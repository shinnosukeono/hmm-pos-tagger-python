import sys

from datasets.treebank_pos import TreebankPOS
from models.hmm import HMM
from preprocess import normalize_number
from trainer import POSAnalysisTrainer
from utils import Compose


def main():
    argv = sys.argv
    if len(argv) > 1:
        root = argv[1]
    else:
        root = None

    preprocessor = Compose([str.lower, normalize_number])

    # build the dataset
    dataset = TreebankPOS(root=root, download=True, trans=preprocessor)

    # create a model
    model = HMM(dataset.n_pos, dataset.n_words)

    # train
    trainer = POSAnalysisTrainer(model=model, dataset=dataset, train_ratio=0.8, shuffle=False)
    trainer.train()

    # evaluation
    trainer.test()


if __name__ == "__main__":
    main()
