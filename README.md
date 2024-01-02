# Usage

```shell
python3 main.py [directory name]
```

- If a directory name is given, the program tries to use the pos files in the directory. The parse implemented here is made for the data in [NLTK Corpora](https://www.nltk.org/nltk_data/).
- If a directory name is not given, the program tries to use the nltk library for loading the data.

## output example

```
loading the library data...
[nltk_data] Downloading package treebank to
[nltk_data]     /Users/xxx/nltk_data...
[nltk_data]   Package treebank is already up-to-date!
the corpus was successfully loaded.
preprocessing the data...
100%|█████████████████████████| 3914/3914 [00:00<00:00, 4746.26it/s]
the number of the unique words: 11541
the number of the unique pos: 47
preprocessing finished.
the size of the train data: 3131
the size of the test data: 783
training...
training finished.
evaluating...
evaluation finished.
Accuracy: 0.9231
```