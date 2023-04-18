"""Provides the data loaders for the datasets that may be used in the pipelines."""

import tarfile
import zipfile

from io import BytesIO
from pathlib import Path

from requests import get


def ds_cloze(path="./data") -> dict[str, Path]:
    """Returns a dataset object for interacting with the cloze test dataset as extracted by XX

    Parameters
    ----------
    path : str, optional
        Default path for storing the files, by default "./data/"

    Returns
    -------
    dict[str,Path]
        A dictionary object with the following structure:
        ```
        - split: path
        ```
        Where source is one of `[test, train, val]` and path is a Path object pointing to the csv file of the dataset.

    Example Usage
    -------
    ```
    import pandas as pd

    ds = ds_cloze()
    df = pd.read_csv(ds["train"])
    df.head()
    ```
    """
    clozepath = Path(path) / "cloze/"

    trainpath = clozepath / "cloze_train.csv"
    testpath = clozepath / "cloze_test.csv"
    valpath = clozepath / "cloze_val.csv"
    if not clozepath.exists():
        clozepath.mkdir(exist_ok=True)
        trainpath.write_bytes(get("https://goo.gl/0OYkPK", timeout=5).content)

        testpath.write_bytes(get("https://goo.gl/BcTtB4", timeout=5).content)

        valpath.write_bytes(get("https://goo.gl/XWjas1", timeout=5).content)

    return {"test": testpath, "train": trainpath, "val": valpath}


def tiny_shakespeare(path="./data/"):
    """p """
    tiny_shakespeare_path = Path(
        path) / "tiny_shakespeare" / "tiny_shakespeare.txt"
    if not tiny_shakespeare_path.exists():
        tiny_shakespeare_path.write_bytes(get(
            "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt", timeout=5).content)

    return tiny_shakespeare_path


def ds_writingprompts(path="./data/") -> dict[str, tuple[Path, Path]]:
    """Returns a dataset object for interacting with the writing prompts dataset as extracted by Fan et al. (2015)

    Returns
    -------
    dict[str, tuple[Path, Path]]
        A dictionary object with the following structure:
        ```
        - split: (source, target)
        ```
        Where source is one of `[test, train, val]` and source and target are the "prompt" and "response(s)" files, respectively.

    Example Usage
    -------
    ```
    ds = ds_writingprompts()
    with open(ds["train"][1]) as f:
        f.read()
    ```
    """
    wppath = Path(path) / "writingPrompts/"
    if not wppath.exists():

        file = tarfile.open(fileobj=BytesIO(get(
            "https://dl.fbaipublicfiles.com/fairseq/data/writingPrompts.tar.gz", timeout=5).content))

        file.extractall(wppath.parent.parent)

    return {"test": (wppath / "test.wp_source", wppath / "test.wptarget"),
            "train": (wppath / "train.wp_source", wppath / "train.wp_target"),
            "val": (wppath / "valid.wp_source", wppath / "valid.wp_target")}


def ds_dgt(path="./data/") -> Path:
    """Returns the DGT-Acquis dataset offered by the European Union, etc.

    Parameters
    ----------
    path : str, optional
        Path to the data directory, by default "./data/"

    Returns
    -------
    Path
        A path reference for the available file that can be loaded next.
    """
    ds_path = Path(path) / "dgt" / "data.en.txt"
    if not ds_path.exists():
        with zipfile.ZipFile(BytesIO(get(
                "https://wt-public.emm4u.eu/Resources/DGT-Acquis-2012/data.en.txt.zip", timeout=5).content)) as file:
            file.extractall(ds_path.parent)

    return ds_path


def load_imageability() -> Path:
    """
    Loads the imageability dataset from Cortese et al. (2004) and returns the path to the file.
    """

    im_path = Path(__package__) / "static" / \
        "imageability" / "cortese20004norms.csv"

    if not im_path.exists():
        with zipfile.ZipFile(BytesIO(get(r'https://static-content.springer.com/esm/art%3A10.3758%2FBF03195585/MediaObjects/Cortese-BRM-2004.zip', timeout=5).content)) as file:
            file.extractall(im_path.parent)

        for file in im_path.parent.glob('**/*'):
            file.rename(im_path.parent / file.name)

        # (im_path.parent / "Cortese-BRMIC-2004").rmdir()
    return im_path

def load_concreteness() -> Path:
    conc_path = Path(__package__) / "static" / \
        "concreteness" / "concreteness.csv"

    if not conc_path.exists():
        conc_path.parent.mkdir(exist_ok=True, parents=True)
        conc_path.write_bytes(get(r'http://crr.ugent.be/papers/Concreteness_ratings_Brysbaert_et_al_BRM.txt', timeout=5).content)

    return conc_path

def load_freq() -> Path:
    """_summary_

    Returns
    -------
    Path
        _description_
    """
    freq_path = Path(__package__) / "static" / \
        "frequency" / "frequency.csv"

    if not freq_path.exists():
        freq_path.parent.mkdir(exist_ok=True, parents=True)
        freq_path.write_bytes(get(r'https://ucrel.lancs.ac.uk/bncfreq/lists/1_1_all_alpha.txt', timeout=5).content)

    return freq_path

if __name__ == "__main__":
    print("You should use the functions defined in the file, not run it directly!")
