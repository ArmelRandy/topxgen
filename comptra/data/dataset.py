from datasets import load_dataset
from comptra.languages import *

def get_datasets(
    dataset_name_or_path: str,
    language: str
):
    """
    Get a dataset given its description and the language of interest
    Arguments
    ---------
        - dataset_name_or_path: str,
            Description of the dataset of interest
        - language: str,
            Language of interest (e.g. English)
    Examples
    --------
    >>> get_datasets("flores", "English")
    DatasetDict({
        dev: Dataset({
            features: ['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence'],
            num_rows: 997
        })
        devtest: Dataset({
            features: ['id', 'URL', 'domain', 'topic', 'has_image', 'has_hyperlink', 'sentence'],
            num_rows: 1012
        })
    })
    """
    if dataset_name_or_path == "flores":
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets
            dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
    elif dataset_name_or_path == "ntrex":
        from comptra.data.ntrex import get_datasets as ntrex
        code = MAPPING_LANG_TO_KEY_NTREX[language]
        dataset = ntrex(code, code)[0]
    elif dataset_name_or_path == "tico":
        from comptra.data.tico import get_datasets as tico
        if language == "English":
            #dataset, _ = tico("English", "Hausa")
            dataset, _ = tico("English", "Bengali")
        else:
            _, dataset = tico("English", language)
    elif dataset_name_or_path == "ood":
        # dev = Flores, devtest = TICO
        from comptra.data.tico import get_datasets as tico
        # FLORES-200
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets
            flores_dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            flores_dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        # TICO-19
        if language == "English":
            #dataset, _ = tico("English", "Hausa")
            dataset, _ = tico("English", "Bengali")
        else:
            _, dataset = tico("English", language)
        # dev = Flores, devtest = TICO
        dataset["dev"] = flores_dataset["dev"]
    elif dataset_name_or_path == "validation":
        if language in NON_FLORES:
            from comptra.data.extension import get_datasets as get_extension_datasets
            dataset = get_extension_datasets(MAPPING_LANG_TO_KEY[language])
        else:
            dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        from datasets import DatasetDict, Dataset
        return DatasetDict(
            {
                "devtest": dataset["dev"],
                "dev": Dataset.from_dict(
                    {
                        col: []
                        for col in dataset["dev"].column_names
                    }
                )
            }
        )
    elif dataset_name_or_path == "topxgen":
        from train_datasets import get
        from datasets import Dataset
        import os
        dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        A = get(
            data_dir=os.path.join(os.path.dirname(__file__), "../..", "data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B"),
            languages=[language],
            percentile=None,
            size=-1,
            strategy="soonest",
            test_size_ratio=0.0,
            seed=122,
            reverse=False
        )
        dataset_src = load_dataset("facebook/flores", "eng_Latn")
        dataset_src["devtest"] = dataset_src["devtest"].remove_columns(["id", "URL", "domain", "topic", "has_image", "has_hyperlink"])
        dataset_src["dev"] = Dataset.from_dict({"sentence": A["source"]})
        
        dataset["devtest"] = dataset["devtest"].remove_columns(["id", "URL", "domain", "topic", "has_image", "has_hyperlink"])
        dataset["dev"] = Dataset.from_dict({"sentence": A["target"]})
        return dataset_src, dataset
    else:
        raise ValueError(f"Unsupported dataset description '{dataset_name_or_path}")
    return dataset

