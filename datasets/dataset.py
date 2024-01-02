from abc import ABC, abstractmethod
from typing import Any


class Dataset(ABC):
    data: list = []

    @abstractmethod
    def __len__(self) -> int:
        return NotImplemented

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        return NotImplemented
