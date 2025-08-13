import os
import json
import numpy as np
from typing import List, Union
from datasets import Dataset, concatenate_datasets


def get(
    data_dir: str,
    languages: List[str],
    percentile: float,
    size: int,
    strategy: str,
    test_size_ratio: Union[float, int],
    seed: int,
    reverse: bool = False,
):
    rng = np.random.default_rng(seed)
    list_of_datasets = []
    for language in languages:
        list_of_propositions = []
        list_of_translations = []
        list_of_scores = []
        with open(
            os.path.join(data_dir, f"{language}_translate_filtered.jsonl"), "r"
        ) as fin:
            for line in fin:
                dico = json.loads(line)
                propositions = dico["propositions"]
                translations = dico["translations"]
                scores = dico["scores"]
                """
                list_of_propositions.extend(propositions)
                list_of_translations.extend(translations)
                list_of_scores.extend(scores)
                """
                for j, (proposition, translation) in enumerate(
                    zip(propositions, translations)
                ):
                    if len(proposition.strip()) < 10 or len(translation.strip()) < 10:
                        continue
                    if any([col in translation for col in ["#", ">"]]) or any(
                        [col in proposition for col in ["#", ">"]]
                    ):
                        continue
                    list_of_propositions.append(proposition)
                    list_of_translations.append(translation)
                    list_of_scores.append(scores[j])

        if percentile is not None:
            threshold = np.percentile(list_of_scores, percentile)
            selected_indices = [
                i for i in range(len(list_of_scores)) if list_of_scores[i] > threshold
            ]
        else:
            if size < 0:
                selected_indices = [i for i in range(len(list_of_propositions))]
            else:
                if strategy == "highest":
                    selected_indices = np.argsort(list_of_scores)[-size:]
                elif strategy == "soonest":
                    selected_indices = [i for i in range(size)]
                elif strategy == "random":
                    selected_indices = rng.choice(
                        a=len(list_of_propositions), size=size, replace=False
                    ).tolist()

        selected_indices = [
            i for i in selected_indices if i < len(list_of_propositions)
        ]
        dataset = Dataset.from_dict(
            {
                "source": [list_of_translations[i] for i in selected_indices],
                "target": [list_of_propositions[i] for i in selected_indices],
                "source_language": ["English"] * len(selected_indices),
                "target_language": [language] * len(selected_indices),
            }
        )
        # dataset = dataset.rename_column("source", input_column_name)
        # dataset = dataset.rename_column("target", output_column_name)
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    if test_size_ratio == 0:
        return dataset
    ds = dataset.train_test_split(test_size=test_size_ratio, shuffle=True, seed=seed)
    if reverse:
        from datasets import DatasetDict

        reverse_ds_train = Dataset.from_dict(
            {
                "source": ds["train"]["target"],
                "target": ds["train"]["source"],
                "source_language": ds["train"]["target_language"],
                "target_language": ds["train"]["source_language"],
            }
        )
        reverse_ds_test = Dataset.from_dict(
            {
                "source": ds["test"]["target"],
                "target": ds["test"]["source"],
                "source_language": ds["test"]["target_language"],
                "target_language": ds["test"]["source_language"],
            }
        )
        print(
            f"We will consider both translation direction: English to {language} and {language} to English."
        )
        return DatasetDict(
            {
                "train": concatenate_datasets([ds["train"], reverse_ds_train]).shuffle(
                    seed=seed
                ),
                "test": concatenate_datasets([ds["test"], reverse_ds_test]).shuffle(
                    seed=seed
                ),
            }
        )
    return ds


from datasets import load_dataset
from comptra.languages import MAPPING_LANG_TO_KEY


def get_flores(
    src: str,
    languages: List[str],
    test_size_ratio: Union[float, int],
    seed: int,
    size: int = None,
):
    list_of_datasets = []
    ds_src = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[src])
    for language in languages:
        ds_tgt = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])
        dataset = Dataset.from_dict(
            {
                "source": ds_src["dev"]["sentence"],
                "target": ds_tgt["dev"]["sentence"],
                "source_language": [src] * len(ds_src["dev"]),
                "target_language": [language] * len(ds_src["dev"]),
            }
        )
        # dataset = dataset.rename_column("source", input_column_name)
        # dataset = dataset.rename_column("target", output_column_name)
        if size is not None and size > 0:
            dataset = dataset.select([i for i in range(size)])
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    ds = dataset.train_test_split(test_size=test_size_ratio, shuffle=True, seed=seed)
    return ds


MP = {
    "Hausa": "ha",
    "Igbo": "ig",
    "Kinyarwanda": "rw",
    "Somali": "so",
    "Swahili": "sw",
    "Xhosa": "xh",
}


def get_smol(
    src: str, languages: List[str], test_size_ratio: Union[float, int], seed: int
):
    assert src == "English"
    list_of_datasets = []
    for language in languages:
        ds_src = load_dataset("google/smol", f"smolsent__en_{MP[language]}")
        dataset = Dataset.from_dict(
            {
                "source": ds_src["train"]["src"],
                "target": ds_src["train"]["trg"],
                "source_language": [src] * len(ds_src["train"]),
                "target_language": [language] * len(ds_src["train"]),
            }
        )
        list_of_datasets.append(dataset)
    dataset = concatenate_datasets(list_of_datasets).shuffle(seed=seed)
    ds = dataset.train_test_split(test_size=test_size_ratio, shuffle=True, seed=seed)
    return ds


if __name__ == "__main__":
    print(get_smol("English", ["Hausa"], 100, 122))
    for T in [0.0, 0.5, 1.0, 1.2]:
        path = os.path.join(os.path.dirname(__file__), f"data/wiki/gemma-3-27b-it/topxgen/T={T}")
        print(f"path: {path}")
        print(
            get(
                data_dir=f"{path}/nllb-200-3.3B",
                #languages=["Sundanese"],
                languages=["Hausa"],
                size=-1,
                percentile=None,
                strategy="soonest",
                test_size_ratio=1000,
                seed=122,
                reverse=True
            )
        )