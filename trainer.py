import numpy as np
from models.model import Model
from utils import Dataloader, Dataset, dataset_split


class POSAnalysisTrainer:
    def __init__(self, model: Model, dataset: Dataset, train_ratio: float, shuffle: bool = False) -> None:
        self.model = model
        dataset_train, dataset_test = dataset_split(dataset, train_ratio, shuffle)
        self.dataloader_train = Dataloader(dataset=dataset_train, batch_size=len(dataset_train), shuffle=False)
        self.dataloader_test = Dataloader(dataset=dataset_test, batch_size=1, shuffle=False)

    def train(self):
        print("training...")
        self.model.train()
        for batch_s, batch_i in self.dataloader_train:
            self.model(batch_s, batch_i)
        print("training finished.")

    def test(self):
        print("evaluating...")
        self.model.eval()
        correct = 0
        n_pred = 0
        for sentence_s, sentence_i in self.dataloader_test:
            _, i_pred = self.model(sentence_s, sentence_i)[0]  # the batch size is 1
            correct += np.sum(sentence_i[0] == i_pred)
            n_pred += len(sentence_i[0])
        print("evaluation finished.")
        print(f"Accuracy: {correct / n_pred:.4f}")
