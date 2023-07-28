import poptorch
import torch

def tensor_remap(x):
    out = poptorch.custom_op(
            [x],
            name="TensorRemap",
            domain="ai.graphcore",
            domain_version=1,
            example_outputs=[x],
            attributes={"remap_type": 0}
        )[0]
    return out

# Attribute:
# enum class TensorRemapType {
# 	  /// Remap the tensor in the forward pass, reverse-apply the remapping in the
# 	  /// backward pass
# 	  FwdBwdReverse = 0,
# 	  /// Remap the tensor in the forward pass and backward pass independently
# 	  FwdBwd,
# 	  /// Only remap the tensor in the forward pass, use identity
# 	  /// for the backward pass
# 	  Fwd
# 	};
