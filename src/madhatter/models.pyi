from nltk.data import find as find
from transformers import AutoTokenizer as AutoTokenizer, BertForMaskedLM as BertForMaskedLM
from typing import Any, Literal, overload

@overload
def sent_predictions(sent: str | list[str], bench: Any, model: Any, tokenizer: Any, return_tokens: Literal[True] = True, k: int = ...) -> tuple[list[list[float]],dict[str, list[str]]] : ...

@overload
def sent_predictions(sent: str | list[str], bench: Any, model: Any,
                     tokenizer: Any, return_tokens: Literal[False], k: int = ...) -> list[list[float]]: ...

@overload
def predict_tokens(sent: str, masked_word: str, model, tokenizer, return_tokens: Literal[False], k: int = ...) -> list[float]: ...

@overload
def predict_tokens(sent: str, masked_word: str, model, tokenizer, return_tokens: Literal[True] = True, k: int = ...) -> list[tuple[str,float]]: ...

