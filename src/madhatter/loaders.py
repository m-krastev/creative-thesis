"""Provides the data loaders for the datasets that may be used in the pipelines."""

import pathlib

import pandas as pd
import requests
from datasets.load import load_dataset

# Check if the file exists, otherwise download it

cloze_train_path = pathlib.Path("./data/cloze_test_2018_train.csv")
if not cloze_train_path.exists():
    cloze_train_path.write_bytes(requests.get(
        "https://goo.gl/0OYkPK", timeout=5).content)

cloze_val_path = pathlib.Path("./data/cloze_test_2018_val.csv")
if not cloze_val_path.exists():
    cloze_val_path.write_bytes(requests.get(
        "https://goo.gl/XWjas1", timeout=5).content)

cloze_test_path = pathlib.Path("./data/cloze_test_2018_test.csv")
if not cloze_test_path.exists():
    cloze_test_path.write_bytes(requests.get(
        "https://goo.gl/BcTtB4", timeout=5).content)

tiny_shakespeare_path = pathlib.Path("./data/tiny_shakespeare.txt")
if not tiny_shakespeare_path.exists():
    tiny_shakespeare_path.write_bytes(requests.get(
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", timeout=5).content)

val_ds = pd.read_csv(cloze_val_path)
# val_ds = load_dataset('csv', "val", data_files=cloze_val_path.__str__())

# tiny_shakespeare = load_dataset('tiny_shakespeare')
with open(tiny_shakespeare_path, mode='r', encoding='utf8') as f:
    tiny_shakespeare = f.read()

# Load the datasets later
train_ds = load_dataset('csv', "train", data_files=cloze_train_path.__str__())
test_ds = load_dataset('csv', "test", data_files=cloze_test_path.__str__())
