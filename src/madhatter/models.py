"""models.py
Base file for LLM operations on text.
"""

# Initialize models
from typing import Any, Literal

import nltk
import torch


def sent_predictions(sent: str | list[str], bench: Any, model: Any, tokenizer: Any, return_tokens: Literal[True,False] = False, k: int = 20):
    """Returns predictions for content words in a given sentence. If return_tokens is true, 
    returns a key-value pair dictionary where the key is the used word, and the value is a list of suggested tokens, 
    corresponding to the likekihoods in the first list.
    """
    if isinstance(sent, str):
        tokens = nltk.word_tokenize(sent.lower())
    elif isinstance(sent, list):
        tokens = [token.lower() for token in sent]
        sent = " ".join(tokens)
    else:
        raise TypeError()
    words = nltk.pos_tag(tokens, tagset='universal')

    results = []
    return_words = {}

    # loop over the words of the sentence
    for word, tag in words:
        # Early stopping
        if word in bench.stopwords or tag not in bench.tags_of_interest:
            continue

        if return_tokens is True:
            toks = predict_tokens(
                sent, word, model, tokenizer, return_tokens=True, k=k)
            predicted_tokens = [_[0] for _ in toks]
            predicted_words = [_[1] for _ in toks]

            results.append(predicted_tokens)
            return_words[(word, tag)] = predicted_words
        else:
            predicted_tokens = predict_tokens(
                sent, word, model, tokenizer, return_tokens=return_tokens, k=k)
            results.append(predicted_tokens)


    return results, return_words if return_tokens is True else results

def sliding_window_preds(sent: str | list[str], bench: Any, model: Any, tokenizer: Any, return_tokens: bool = False, k: int = 20, n: int = 30):
    """Returns predictions for content words in a given sliding window of length n=30. If return_tokens is true, 
    returns a key-value pair dictionary where the key is the used word, and the value is a list of suggested tokens, 
    corresponding to the likekihoods in the first list.
    """
    if isinstance(sent, str):
        tokens = nltk.word_tokenize(sent.lower())
    elif isinstance(sent, list):
        tokens = [token.lower() for token in sent]
        sent = " ".join(tokens)
    else:
        raise TypeError()

    # TODO: Implement a second function working with already postagged sents
    words = nltk.pos_tag(tokens, tagset='universal')

    if len(tokens) < n:
        raise ValueError(f'The given window ({len(tokens)=}) contains less tokens than the requested sliding window({n=}); ')
    results = []
    return_words = {}


    # for i in range(len(words)):
        # for j in range(n)

    # loop over the words of the sentence
    for word, tag in words:
        # Early stopping
        if tag not in bench.tags_of_interest or word in bench.stopwords:
            continue

        if return_tokens is True:
            toks = predict_tokens(
                sent, word, model, tokenizer, return_tokens=return_tokens, k=k)
            predicted_tokens, predicted_words = list(zip(*toks))

            results.append(predicted_tokens)
            return_words[(word, tag)] = predicted_words
        else:
            predicted_tokens = predict_tokens(
                sent, word, model, tokenizer, return_tokens=return_tokens, k=k)
            results.append(predicted_tokens)

    return results, return_words if return_tokens is True else results

def predict_tokens(sent: str, masked_word: str, model, tokenizer, return_tokens: bool = False, k: int = 20) -> list[float]|list[tuple[str,float]]:
    """Predict the top k tokens that could replace the masked word in the sentence. 

    Returns a list of tuples of the form (token, likelihood, similarity) where similarity is the cosine similarity of the given words in a word2vec model.

    Parameters
    ----------
    sent: str
        The sentence to predict tokens for.
    masked_word: str
        The word to predict tokens for. Note that this word must be in the sentence.
    model
        Must be a masked language model that takes in a sentence and returns a tensor of logits for each token
        in the sentence. Default assumes a pretrained BERT model from the HuggingFace `transformers` library.
    word2vec_model
        Must be a word2vec model that takes in a word and returns a vector representation of the word.
        Default is `gensim.models.keyedvectors.KeyedVectors` loaded from the `word2vec_sample` model 
        from the `nltk_data` package.
    k: int
        The number of tokens to return.    

    Returns
    -------
    List of tuples the form (token, likelihood, similarity)

    token: str
        The predicted token.
    likelihood: float
        The likelihood of the token being the masked word.
    similarity: float
        The cosine similarity of the token and the masked word.
    """
    if masked_word not in sent:
        raise ValueError(f"{masked_word} not in {sent}")
    masked_sent = sent.replace(masked_word, "[MASK]")

    inputs = tokenizer(masked_sent, return_tensors="pt")

    with torch.no_grad():
        logits = model(**inputs).logits

    # retrieve index of [MASK]
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[
        0].nonzero(as_tuple=True)[0]

    vals, predicted_token_ids = torch.topk(logits[0, mask_token_index], k, dim=-1) # pylint: disable=no-member

    ret = []
    for i, predicted_token_id in enumerate(predicted_token_ids[0]):
        # if the actual tokens are needed, return those as well
        if return_tokens is True:
            word = tokenizer.decode(predicted_token_id)

            # If word is a subword, combine it with the previous word
            word = word if not word.startswith("##") else masked_word+word[2:]

            ret.append((word, vals[0, i].item()))
        else:
            ret.append(vals[0,i].item())

    return ret
