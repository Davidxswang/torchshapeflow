from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    """Tensor shape annotation for use with ``typing.Annotated``.

    Each positional argument describes one dimension: a ``str`` names a symbolic
    dimension (e.g. a batch size), while an ``int`` gives a fixed size.

    Example::

        from typing import Annotated
        import torch
        from torchshapeflow import Shape

        def forward(x: Annotated[torch.Tensor, Shape("B", "T", 768)]) -> ...:
            ...

    The frozen dataclass uses a custom ``__init__`` so that ``*dims`` varargs are
    stored as a tuple field, which is required because ``frozen=True`` normally
    prevents attribute assignment after ``__init__``.
    """

    dims: tuple[str | int, ...]

    def __init__(self, *dims: str | int) -> None:
        object.__setattr__(self, "dims", tuple(dims))

    def __iter__(self) -> Iterator[str | int]:
        return iter(self.dims)
