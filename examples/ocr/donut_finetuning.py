import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import DonutProcessor

from examples.ocr.donut_model import IPUDonutModel
from optimum.graphcore import IPUConfig, IPUSeq2SeqTrainer, IPUSeq2SeqTrainingArguments
from optimum.utils import logging


logger = logging.get_logger(__name__)


# define paths
base_path = Path("examples/ocr/data")
metadata_path = base_path.joinpath("key")
image_path = base_path.joinpath("img")

# Load dataset
dataset = load_dataset("imagefolder", data_dir=image_path, split="train[:16]")

new_special_tokens = []  # new tokens which will be added to the tokenizer
task_start_token = "<s>"  # start of task token
eos_token = "</s>"  # eos token of tokenizer


def json2token(obj, update_special_tokens_for_json_key: bool = True, sort_json_key: bool = True):
    """
    Convert an ordered JSON object into a token sequence
    """
    if type(obj) == dict:
        if len(obj) == 1 and "text_sequence" in obj:
            return obj["text_sequence"]
        else:
            output = ""
            if sort_json_key:
                keys = sorted(obj.keys(), reverse=True)
            else:
                keys = obj.keys()
            for k in keys:
                if update_special_tokens_for_json_key:
                    new_special_tokens.append(rf"<s_{k}>") if rf"<s_{k}>" not in new_special_tokens else None
                    new_special_tokens.append(rf"</s_{k}>") if rf"</s_{k}>" not in new_special_tokens else None
                output += (
                    rf"<s_{k}>" + json2token(obj[k], update_special_tokens_for_json_key, sort_json_key) + rf"</s_{k}>"
                )
            return output
    elif type(obj) == list:
        return r"<sep/>".join([json2token(item, update_special_tokens_for_json_key, sort_json_key) for item in obj])
    else:
        # excluded special tokens for now
        obj = str(obj)
        if f"<{obj}/>" in new_special_tokens:
            obj = f"<{obj}/>"  # for categorical special tokens
        return obj


def preprocess_documents_for_donut(sample):
    # create Donut-style input
    text = json.loads(sample["text"])
    d_doc = task_start_token + json2token(text) + eos_token
    # convert all images to RGBmap
    image = sample["image"].convert("RGB")
    return {"image": image, "text": d_doc}


torch.set_num_threads(1)
proc_dataset = dataset.map(preprocess_documents_for_donut, num_proc=4)
torch.set_num_threads(4)

processor = DonutProcessor.from_pretrained("philschmid/donut-base-sroie")

# add new special tokens to tokenizer
processor.tokenizer.add_special_tokens(
    {"additional_special_tokens": new_special_tokens + [task_start_token] + [eos_token]}
)

# we update some settings which differ from pretraining; namely the size of the images + no rotation required
# resizing the image to smaller sizes from [1920, 2560] to [960,1280]
processor.feature_extractor.size = [720, 960]  # should be (width, height)
processor.feature_extractor.do_align_long_axis = False


def transform_and_tokenize(sample, processor=processor, split="train", max_length=512, ignore_id=-100):
    # create tensor from image
    try:
        pixel_values = (
            processor(sample["image"], random_padding=split == "train", return_tensors="pt")
            .pixel_values.squeeze()
            .half()
        )
    except Exception as e:
        print(sample)
        print(f"Error: {e}")
        return {}

    # tokenize document
    input_ids = processor.tokenizer(
        sample["text"],
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )["input_ids"].squeeze(0)

    labels = input_ids.clone()
    labels[labels == processor.tokenizer.pad_token_id] = ignore_id  # model doesn't need to predict pad token
    return {"pixel_values": pixel_values, "labels": labels, "target_sequence": sample["text"]}


# need at least 32-64GB of RAM to run this
torch.set_num_threads(1)
processed_dataset = proc_dataset.map(transform_and_tokenize, remove_columns=["image", "text"], num_proc=4)
torch.set_num_threads(4)

# processed_dataset = processed_dataset.train_test_split(test_size=0.1)

# Load model from huggingface.co
model = IPUDonutModel.from_pretrained("naver-clova-ix/donut-base")

# Resize embedding layer to match vocabulary size
# NB: add two here to make it divisible by 5 -> 57535
new_emb = model.decoder.resize_token_embeddings(len(processor.tokenizer) + 2)
print(f"New embedding size: {new_emb}")
# Adjust our image size and output sequence lengths
model.config.encoder.image_size = processor.feature_extractor.size[::-1]  # (height, width)
model.config.decoder.max_length = len(max(processed_dataset["labels"], key=len))

# Add task token for decoder to start
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(["<s>"])[0]

model.train()

ipu_config = IPUConfig.from_dict(
    {
        "optimizer_state_offchip": True,
        "recompute_checkpoint_every_layer": True,
        "enable_half_partials": True,
        "executable_cache_dir": "./exe_cache",
        "gradient_accumulation_steps": 16,
        "replication_factor": 1,
        "ipus_per_replica": 4,
        # "layers_per_ipu": [5, 7, 5, 7],
        "matmul_proportion": [0.2, 0.2, 0.2, 0.2],
        "projection_serialization_factor": 5,
        # "inference_replication_factor": 1,
        # "inference_layers_per_ipu": [12, 12],
        # "inference_parallelize_kwargs": {
        #     "use_cache": True,
        #     "use_encoder_output_buffer": True,
        #     "on_device_generation_steps": 16,
        # }
    }
)

training_args = IPUSeq2SeqTrainingArguments(
    output_dir="./donut-checkpoints",
    do_train=True,
    do_eval=False,
    evaluation_strategy="no",
    predict_with_generate=True,
    num_train_epochs=3,
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    weight_decay=0.01,
    save_total_limit=1,
    save_strategy="epoch",
    logging_steps=100,
    # push to hub parameters
    # report_to="tensorboard",
    # push_to_hub=True,
    # hub_strategy="every_save",
    # hub_model_id=hf_repository_id,
    # hub_token=HfFolder.get_token(),
)

# Create Trainer
trainer = IPUSeq2SeqTrainer(
    model=model,
    ipu_config=ipu_config,
    args=training_args,
    train_dataset=processed_dataset,
)

trainer.train()
