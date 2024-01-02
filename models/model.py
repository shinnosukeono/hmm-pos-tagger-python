from abc import ABC, abstractmethod
from typing import Any


class Model(ABC):
    def __call__(self, *args: Any) -> Any:
        return self.forward(*args)

    @abstractmethod
    def train(self):
        return NotImplemented

    @abstractmethod
    def eval(self):
        return NotImplemented

    @abstractmethod
    def forward(self, *args: Any) -> Any:
        return NotImplemented
