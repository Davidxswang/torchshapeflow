"""Module-level lookup tables shared by the analyzer's walker modules."""

from __future__ import annotations

from torchshapeflow.model import (
    Conv2dSpec,
    CustomModuleSpec,
    EmbeddingSpec,
    LinearSpec,
    LSTMSpec,
    MultiheadAttentionSpec,
    PassthroughSpec,
    Pool2dSpec,
    RepeatSpec,
    SequentialSpec,
)

# nn.Module types whose output shape equals their input shape.
PASSTHROUGH_SUFFIXES: frozenset[str] = frozenset(
    {
        "BatchNorm1d",
        "BatchNorm2d",
        "BatchNorm3d",
        "LayerNorm",
        "Dropout",
        "Dropout2d",
        "Dropout3d",
        "ReLU",
        "LeakyReLU",
        "GELU",
        "SiLU",
        "Sigmoid",
        "Tanh",
        "ELU",
        "SELU",
        "PReLU",
        "Mish",
        "Hardswish",
        "Hardsigmoid",
        "Identity",
        "Softmax",
    }
)

# Reduction ops recognized on both tensors (x.sum) and torch.* functions (torch.sum).
REDUCTION_OPS: frozenset[str] = frozenset(
    {
        "sum",
        "mean",
        "max",
        "min",
        "amax",
        "amin",
        "prod",
        "all",
        "any",
        "argmax",
        "argmin",
        "nanmean",
        "nansum",
    }
)

# Tensor methods that preserve shape (dtype/device casts, memory management, etc.)
PASSTHROUGH_METHODS: frozenset[str] = frozenset(
    {
        "contiguous",
        "float",
        "half",
        "double",
        "int",
        "long",
        "short",
        "byte",
        "bool",
        "to",
        "detach",
        "clone",
        "cpu",
        "cuda",
        "type",
        "masked_fill",
        "masked_fill_",
        "requires_grad_",
        "fill_",
        "zero_",
        "normal_",
        "uniform_",
        "flip",
        "abs",
        "neg",
        "sign",
    }
)

# Functional API suffixes whose output shape equals the first argument's shape.
FUNCTIONAL_PASSTHROUGH: frozenset[str] = frozenset(
    {
        "softmax",
        "log_softmax",
        "relu",
        "relu_",
        "leaky_relu",
        "leaky_relu_",
        "gelu",
        "silu",
        "sigmoid",
        "tanh",
        "elu",
        "selu",
        "mish",
        "hardswish",
        "dropout",
        "dropout2d",
        "dropout3d",
        "layer_norm",
        "batch_norm",
        "group_norm",
        "instance_norm",
        "normalize",
        "triu",
        "tril",
        "flip",
        "isfinite",
        "isinf",
        "isnan",
        "abs",
        "neg",
        "sign",
    }
)

# *_like constructors: output shape equals first argument's shape.
LIKE_OPS: frozenset[str] = frozenset(
    {"zeros_like", "ones_like", "empty_like", "full_like", "rand_like", "randn_like"}
)

# Size-based constructors: shape is built from positional/keyword size args.
TENSOR_CONSTRUCTORS: frozenset[str] = frozenset({"zeros", "ones", "empty", "randn", "rand", "full"})

# Tensor methods that return non-tensor values (no shape to track).
# These should NOT trigger TSF2001 when they return None from _eval_tensor_method.
NON_TENSOR_METHODS: frozenset[str] = frozenset({"item", "numpy", "tolist", "dim"})

# Python builtins and common utility functions that should NOT trigger TSF2002.
BUILTIN_NAMES: frozenset[str] = frozenset(
    {
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "int",
        "float",
        "str",
        "list",
        "tuple",
        "dict",
        "set",
        "type",
        "isinstance",
        "hasattr",
        "getattr",
        "sorted",
        "reversed",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "map",
        "filter",
        "any",
        "all",
        "bool",
        "id",
        "repr",
        "hash",
        "iter",
        "next",
        "super",
        "vars",
        "dir",
        "object",
        "property",
        "staticmethod",
        "classmethod",
    }
)

MODULE_SPEC_TYPES = (
    LinearSpec,
    Conv2dSpec,
    PassthroughSpec,
    EmbeddingSpec,
    Pool2dSpec,
    CustomModuleSpec,
    RepeatSpec,
    SequentialSpec,
    MultiheadAttentionSpec,
    LSTMSpec,
)
