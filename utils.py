import copy
import random
from typing import Any, Callable, Iterator

from datasets.dataset import Dataset


class Dataloader(Iterator):
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool = False) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(self.dataset)))

    def __iter__(self) -> Iterator:
        if self.shuffle:
            random.shuffle(self.indices)
        self.current = 0
        return self

    def __next__(self) -> Any:
        if self.current > len(self.dataset) - self.batch_size:
            raise StopIteration

        indices = self.indices[self.current : self.current + self.batch_size]
        batch_s, batch_i = [], []
        for i in indices:
            s, i = self.dataset[i]
            batch_s.append(s)
            batch_i.append(i)
        self.current += self.batch_size
        return batch_s, batch_i


class Compose:
    def __init__(self, trans_list: list[Callable]) -> None:
        self.trans_list = trans_list

    def __call__(self, x: Any) -> Any:
        for trans in self.trans_list:
            x = trans(x)
        return x


def dataset_split(dataset: Dataset, train_ratio: float, shuffle: bool = False) -> tuple[Dataset, Dataset]:
    data_copied = copy.deepcopy(dataset.data)
    if shuffle:
        random.shuffle(data_copied)

    idx = int(len(data_copied) * train_ratio)
    train_data, test_data = data_copied[:idx], data_copied[idx:]
    train_dataset, test_dataset = copy.deepcopy(dataset), copy.deepcopy(dataset)
    train_dataset.data = train_data
    test_dataset.data = test_data
    print(f"the size of the train data: {len(train_dataset.data)}")
    print(f"the size of the test data: {len(test_dataset.data)}")
    return train_dataset, test_dataset
