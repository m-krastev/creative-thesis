from madhatter.typings.utils import _ratings


import pandas as pd


def concreteness(data: str | list[str], concreteness_df: pd.DataFrame) -> float | None | list[float | None]:
    """Returns the mean concreteness rating for a given word or list of words, according to the table of ~40,000 words and word definitions, as defined by Brysbaert et al (2013)."""
    # TODO: Possibly look at amortized values given standard deviations

    # Fastest way for lookups so far.
    conc = dict(
        zip(concreteness_df["Word"], concreteness_df["Conc.M"]))

    return _ratings(data, conc)