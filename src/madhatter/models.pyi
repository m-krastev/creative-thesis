from nltk.data import find as find
from transformers import AutoTokenizer as AutoTokenizer, BertForMaskedLM as BertForMaskedLM
from typing import Any, Literal, overload

@overload
def sent_predictions(sent: str | list[str], bench: Any, model: Any, tokenizer: Any, return_tokens: bool = True, k: int = ...) -> tuple[list[list],dict[str, list[str]]] : ...

@overload
def sent_predictions(sent: str | list[str], bench: Any, model: Any,
                     tokenizer: Any, return_tokens: bool = False, k: int = ...) -> list[list]: ...

@overload
def predict_tokens(sent: str, masked_word: str, model, tokenizer, return_tokens: Literal[False] = False, k: int = ...) -> list[float]: ...

@overload
def predict_tokens(sent: str, masked_word: str, model, tokenizer, return_tokens: Literal[True] = True, k: int = ...) -> list[tuple[str,float]]: ...

