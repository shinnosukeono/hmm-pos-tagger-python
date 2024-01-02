from collections import Counter
from pathlib import Path
from typing import Any, Callable

import nltk
import numpy as np
from const import POS, Words
from nltk.corpus import treebank
from tqdm import tqdm

from .dataset import Dataset


class TreebankPOS(Dataset):
    def __init__(self, root: str | None = None, download: bool = True, trans: Callable | None = None) -> None:
        self.trans = trans

        if root is None:
            if download:
                print("loading the library data...")
                nltk.download("treebank")
                self.data = treebank.tagged_sents()
            else:
                raise RuntimeError("Thedataset does not exist in thedirectory.")
        else:
            path_root = Path().absolute().joinpath(root)

            if path_root.exists():
                self.data = []
                print("loading the local data ...")
                for filepath in tqdm(path_root.glob("*.pos")):
                    self.data += self._parse_pos_file(filepath)
            else:
                if download:
                    print("loading the library data...")
                    nltk.download("treebank")
                    self.data = treebank.tagged_sents()
                else:
                    raise RuntimeError("The dataset does not exist in the directory.")

        print("the corpus was successfully loaded.")
        print("preprocessing the data...")
        pos_set = set()
        word_list = []
        for sentence in tqdm(self.data):
            words, pos = zip(*sentence)

            if self.trans:
                words = map(self.trans, words)

            word_list += words
            pos_set.update(pos)

        self.counter = Counter(word_list)
        word_set = set(word_list)
        self.n_words = len(word_set) + 1  # +1 for the "UNK"
        print(f"the number of the unique words: {self.n_words}")
        self.word_to_num = {word: i for i, word in enumerate(word_set)}
        self.word_to_num[Words.UNK_KEY] = self.n_words - 1
        self.n_pos = len(pos_set) + 1  # +1 for BOS
        print(f"the number of the unique pos: {self.n_pos}")
        self.pos_to_num = {pos: i + 1 for i, pos in enumerate(pos_set)}
        self.pos_to_num[POS.BOS_KEY] = POS.BOS_VALUE
        print("preprocessing finished.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: slice | int) -> Any:
        if isinstance(idx, slice):
            s_list, i_list = [], []
            for sentence in self.data[idx]:
                s, i = zip(*sentence)

                if self.trans:
                    s = map(self.trans, s)

                s = [self.word_to_num[self._process_unknown(word)] for word in s]
                i = [self.pos_to_num[pos] for pos in i]
                s_list.append(np.array(s))
                i_list.append(np.array(i))
            return s_list, i_list
        else:
            sentence = self.data[idx]
            s, i = zip(*sentence)

            if self.trans:
                s = map(self.trans, s)

            s = [self.word_to_num[self._process_unknown(word)] for word in s]
            i = [self.pos_to_num[pos] for pos in i]
            return np.array(s), np.array(i)

    def _process_unknown(self, word: str) -> str:
        if self.counter[word] <= 1:
            return Words.UNK_KEY
        else:
            return word

    def _parse_pos_file(self, filepath: Path) -> list:
        with open(filepath, "r", encoding="utf-8") as file:
            lines = file.readlines()

        sentences = []
        current_sentence = []

        for line in lines:
            # detect the beginning of a line
            if line.strip() in ["======================================", ""]:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            else:
                # ignore "[]"
                line = line.replace("[", "").replace("]", "")
                words = line.split()

                for word in words:
                    if "/" in word:
                        # split by the last "/"
                        split_index = word.rfind("/")
                        word_text = word[:split_index]
                        word_tag = word[split_index + 1 :]
                        current_sentence.append((word_text, word_tag))

        # add the last line
        if current_sentence:
            sentences.append(current_sentence)

        return sentences
