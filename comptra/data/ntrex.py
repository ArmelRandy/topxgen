from datasets import load_dataset, Dataset, DatasetDict

ntrex = load_dataset("mteb/NTREX", "default")

def get_datasets(
    src: str,
    tgt: str,
    test_samples: int = 1000
    ):
    """
    Arguments
    ---------
        - src: str,
            Source language e.g. eng_Latn
        - tgt: str,
            Target language e.g. fra_Latn
        - test_samples: int,
    Examples
    --------
    >>> get_datasets("eng_Latn", "fra_Latn")
    """
    ds_src = DatasetDict(
        {
            "devtest" : Dataset.from_dict(
                {
                    "sentence" : ntrex["test"][src][:test_samples]
                }
            ),
            "dev": Dataset.from_dict(
                {
                    "sentence": ntrex["test"][src][test_samples:]
                }
            )
        }
    )

    ds_tgt = DatasetDict(
        {
            "devtest" : Dataset.from_dict(
                {
                    "sentence" : ntrex["test"][tgt][:test_samples]
                }
            ),
            "dev": Dataset.from_dict(
                {
                    "sentence": ntrex["test"][tgt][test_samples:]
                }
            )
        }
    )

    return ds_src, ds_tgt