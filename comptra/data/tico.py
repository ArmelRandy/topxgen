from datasets import load_dataset, DatasetDict, Dataset

MP = {
    "Amharic": "am",
    "Arabic": "ar",
    "Bengali": "bn",
    "Kurdish Sorani": "ckb",
    "Latin American Spanish": "es-LA",
    "Farsi": "fa",
    "French": "fr",
    "Nigerian fufulde": "fuv",
    "Hausa": "ha",
    "Hindi": "hi",
    "Indonesian": "id",
    "Kurdish Kurmanji": "ku",
    "Lingala": "ln",
    "Luganda": "lg",
    "Marathi": "mr",
    "Malay": "ms",
    "Muanmar": "my",
    "Nepali": "ne",
    "Oromo": "om",
    "Dari": "prs",
    "Pashto": "ps",
    "Brazilian Portuguese": "pt-BR",
    "Russian": "ru",
    "Kinyarwanda": "rw",
    "Somali": "so",
    "Swahili": "sw",
    "Ethiopian Tigrinya": "ti",
    "Tagalog": "tl",
    "Urdu": "ur",
    "Chinese (Simplified)": "zh",
    "Zulu": "zu",
    "Dinka": "din",
    "Khmer": "km",
    "Kanuri": "kr",
    #"nus": "",
    "Tamil": "ta",
    "English": "en"
}

def get_datasets(src, tgt):
    """
    Arguments
    ---------
        - src: str,
            Source language e.g. English
        - tgt: str,
            Target language e.g. Hausa
        - test_samples: int,
    Examples
    --------
    >>> get_datasets("English", "Hausa")
    """
    print(f"go: {MP[src]}-{MP[tgt]}")
    ds = load_dataset("gmnlp/tico19", f"{MP[src]}-{MP[tgt]}")

    ds_src = DatasetDict(
        {
            "dev": Dataset.from_dict(
                {
                    "sentence": [example["sourceString"].strip() for example in ds["validation"]]
                }
            ),
            "devtest": Dataset.from_dict(
                {
                    "sentence": [example["sourceString"].strip() for example in ds["test"]]
                }
            )
        }
    )

    ds_tgt = DatasetDict(
        {
            "dev": Dataset.from_dict(
                {
                    "sentence": [example["targetString"].strip() for example in ds["validation"]]
                }
            ),
            "devtest": Dataset.from_dict(
                {
                    "sentence": [example["targetString"].strip() for example in ds["test"]]
                }
            )
        }
    )
    return ds_src, ds_tgt 