# Initialize models
import torch
from transformers import AutoTokenizer, BertForMaskedLM
import gensim
from nltk.data import find
word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
    str(find('models/word2vec_sample/pruned.word2vec.txt')), binary=False)

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForMaskedLM.from_pretrained("bert-base-uncased")
