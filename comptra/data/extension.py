from datasets import DatasetDict, Dataset
from comptra.data import DATA_PATH
import os


def get_datasets(code: str):
    """
    Get FLORES 200 data for additional languages
    Arguments
    ---------
    - code: str,
        Code representing the language e.g. nqo_Nkoo for the N'ko Language
    """
    dev = (
        open(os.path.join(DATA_PATH, f"flores/dev/{code}.dev"), "r")
        .read()
        .strip()
        .split("\n")
    )
    devtest = (
        open(os.path.join(DATA_PATH, f"flores/devtest/{code}.devtest"), "r")
        .read()
        .strip()
        .split("\n")
    )

    dataset_tgt = DatasetDict(
        {
            "dev": Dataset.from_dict({"sentence": dev}),
            "devtest": Dataset.from_dict({"sentence": devtest}),
        }
    )
    return dataset_tgt
