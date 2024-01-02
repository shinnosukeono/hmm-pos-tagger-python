from dataclasses import dataclass


@dataclass
class POS:
    BOS_KEY: str = "BOS"
    BOS_VALUE: int = 0


@dataclass
class Words:
    UNK_KEY: str = "UNKNOWN_WORD_IN_TRAINING_DATA"
