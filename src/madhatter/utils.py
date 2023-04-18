from io import BytesIO
import pkgutil
from typing import Literal, Optional, Sequence
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


def get_imageability_df(format="df") -> pd.DataFrame | dict:
    """Returns a table of the imageability of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""

    # load_imageability()
    # Dicts are the fastest way to make string accesses
    dataframe = pd.read_csv(BytesIO(pkgutil.get_data(__package__, "static/imageability/cortese2004norms.csv")), header=9) # type: ignore
    return dataframe if format == "df" else dict(
        zip(dataframe["item"], dataframe["rating"]))



def imageability(data: str | list[str], imageability_df: pd.DataFrame) -> Optional[float] | list[Optional[float]]:
    """Returns the mean imageability rating for a given word or list of words, according to the table of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""
    # TODO: Possibly look at amortized values given standard deviations

    # Fastest way for lookups so far.
    dictionary = dict(
        zip(imageability_df["item"], imageability_df["rating"]))

    return _ratings(data, dictionary)

def _ratings(data: str | list[str], func: dict) -> float | None | list[float | None]:
    """j"""

    if isinstance(data, str):
        return func.get(data.lower(), None)
    if isinstance(data, list):
        return [func.get(w.lower(), None) for w in data if w not in stopwords] # type: ignore

    raise TypeError(
        f"Inappropriate argument type for `word`. Expected `list` or `str`, but got {type(data)}")


def get_freq_df(_format: Literal["df"] | Literal["dict"] = "df") -> pd.DataFrame | dict:
    '''
    Key:
    
    Word = Word type (headword followed by any variant forms) - see pp.4-5
    
    PoS  = Part of speech (grammatical word class - see pp. 12-13)
    
    Freq = Rounded frequency per million word tokens (down to a minimum of 10 occurrences of a lemma per million)- see pp. 5
    
    Ra   = Range: number of sectors of the corpus (out of a maximum of 100) in which the word occurs
    
    Disp = Dispersion value (Juilland's D) from a minimum of 0.00 to a maximum of 1.00.
    '''

    df_freq = pd.read_csv(BytesIO(pkgutil.get_data(__package__, "static/frequency/frequency.csv")), encoding='unicode_escape', sep="\t") # type: ignore

    # drop unnamed column one cuz trash source,
    # also remove the column storing word variants
    df_freq = df_freq.drop([df_freq.columns[0], df_freq.columns[3]], axis=1)

    # clean the data
    df_freq = df_freq.convert_dtypes()
    df_freq["Freq"] = pd.to_numeric(df_freq["Freq"], errors="coerce")
    df_freq["Ra"] = pd.to_numeric(df_freq["Ra"], errors="coerce")
    df_freq["Disp"] = pd.to_numeric(df_freq["Disp"], errors="coerce")
    df_freq = df_freq.dropna()

    # filter out the word variants
    df_freq = df_freq.loc[df_freq["Word"] != '@']

    # replace the PoS tags with the ones we are using
    def replace_df(df, column, replacements) -> pd.DataFrame:
        for src, tar in replacements:
            df.loc[df[column].str.contains(src), column] = tar
        return df

    replacements = [("No", "NOUN"), ("Adv", "ADV"),
                    ("Adj", "ADJ"), ("Verb", "VERB")]
    df_freq = replace_df(df_freq, "PoS", replacements)

    if _format == "df":
        return df_freq

    # # set the index to the "Word" column so lookups are faster
    # df = df.set_index('Word')

    # I don't particularly care enough for disambiguating their PoS tags :skull:, so might as well aggregate the columns and make it even faster.
    # group everything together because i literally cant bother with pos tag lookups on big scales
    df_freq = df_freq.groupby('Word').sum(numeric_only=True)

    # df_freq_dict = dict(zip(((x,y) for x, y in zip(df_freq.index, df_freq["PoS"]) if y in TAGS_OF_INTEREST), df_freq["Freq"])) # 5600 entries

    return dict(zip(df_freq.index, -np.log10(df_freq["Freq"])))  # 5900 entries


def frequency_ratings(data):
    """Returns log10 frequency for lemmatized words"""
    df_dict = get_freq_df("dict")  # 6652 entries

    return _ratings(data, df_dict) # type: ignore
