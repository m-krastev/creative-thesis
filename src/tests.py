# %%
import spacy
from nltk.corpus import gutenberg

import madhatter as md
from madhatter.benchmark import CreativityBenchmark

# %%
files = [gutenberg.raw(fileid)[:100_000] for fileid in gutenberg.fileids()]
files = files*30

# nlp = spacy.load("en_core_web_sm", disable=[
#                  "ner", 
#                 #  "lemmatizer", 
#                  "textcat", "attribute_ruler"])


def process(file):
    bench = CreativityBenchmark(file)
    sent = bench.tagged_sents[0]
    for tup in sent:
        pass
    del bench

if __name__ == '__main__':
    # docs = nlp.pipe(files, n_process=8)
    # for doc in docs:
    #     # gc.collect()
    #     sent = list(doc.sents)[0]
    #     for token in sent:
    #         # print(token.text, token.tag_, token.pos_, token.dep_, token.is_stop)
    #         pass

    import multiprocessing as mp
    
    with mp.Pool(8) as p:
        l = p.map(process, files)



# %% [markdown]
# ### Memory usage of Spacy vs Custom Package
# | Framework | peak memory | increment |
# |  ------ | ---------- | -------- |
# | Spacy | 5089.13 MiB |  4465.29 MiB |
# | Mad Hatter| 434.81 MiB  | 48.75 MiB |
# 
# Increment here is the more important number as it tells us how memory usage peaks when performing a given operation.
# 


