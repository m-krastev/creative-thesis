from io import BytesIO
import pkgutil
from typing import Sequence
import numpy.typing as ntp
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
# from .loaders import load_ concreteness, load_imageability

stopwords = set(stopwords.words('english'))


def mean(items: Sequence) -> float:
    return 0 if len(items) == 0 else sum(items)/len(items)


def cross_softmax(one: ntp.NDArray, two: ntp.NDArray, temp1=0.5, temp2=0.5):
    # return (torch.softmax(torch.from_numpy(results[1][:,0]), dim=0) @ torch.softmax(torch.from_numpy(results[1][:,1]), dim=0)).item()
    exps = np.exp(one*temp1)
    exps /= exps.sum()
    exps2 = np.exp(two*temp2)
    exps2 /= exps2.sum()
    return exps @ exps2


def slope_coefficient(one: ntp.NDArray, two: ntp.NDArray) -> float:
    """Returns the coefficient of the slope"""
    # Using the integrated function
    # return np.tanh(np.polyfit(x,y,1)[0])
    # Manually implementing slope equation
    return ((one*two).mean(axis=0) - one.mean()*two.mean(axis=0)) / ((one**2).mean() - (one.mean())**2)


def get_concreteness_df() -> pd.DataFrame:
    # load_concreteness()
    return pd.read_csv(BytesIO(pkgutil.get_data(__package__, 'static/concreteness/concreteness.csv')), sep="\t") # type: ignore

    # concreteness_df = concreteness_df.set_index("Word").sort_index()


def get_imageability_df() -> pd.DataFrame:
    # load_imageability()
    return pd.read_csv(BytesIO(pkgutil.get_data(__package__, "static/imageability/cortese2004norms.csv")), header=9) # type: ignore


def imageability(data: str | list[str], imageability_df: pd.DataFrame) -> float | None | list[float | None]:
    """Returns the mean imageability rating for a given word or list of words, according to the table of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""
    # TODO: Possibly look at amortized values given standard deviations

    # Fastest way for lookups so far.
    imgeability = dict(
        zip(imageability_df["item"], imageability_df["rating"]))

    if isinstance(data, str):
        return imgeability.get(data.lower(), None)
    if isinstance(data, list):
        return [imgeability.get(w.lower(), None) for w in data if w not in stopwords] # type: ignore

    raise TypeError(
        f"Inappropriate argument type for `word`. Expected `list` or `str`, but got {type(data)}")


def rare_word_usage(data: str | list[str]):
    pass
