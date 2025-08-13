import argparse
import difflib
import re
import unicodedata
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_dataset, Dataset

from comptra.data.dataset import get_datasets
from datasets import concatenate_datasets
import json
import os

from sentence_splitter import SentenceSplitter

splitter = SentenceSplitter(language="en")

PATH = os.path.dirname(__file__)

def tokenize(text):
    """Normalize text by removing diacritics and tokenize."""
    text = "".join(
        c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn"
    )
    tokens = re.findall("\w+", text.lower())
    return tokens


def get_ngrams(tokens, n):
    """Generate n-grams from tokens."""
    return set(zip(*[tokens[i:] for i in range(n)]))


def retrieve_ngrams_batch(batch, eval_ngrams, eval_datasets, eval_texts, ngram_len):
    """Find contaminated samples based on n-grams."""
    new_batch = {"completion": [], "ngram": [], "bench_name": [], "bench_text": []}
    for completion in batch["completion"]:
        tokens = tokenize(completion)
        ngrams = get_ngrams(tokens, ngram_len)
        for ngram in ngrams:
            if ngram in eval_ngrams:
                idx = eval_ngrams[ngram]
                new_batch["completion"].append(completion)
                new_batch["ngram"].append(ngram)
                new_batch["bench_name"].append(eval_datasets[idx])
                new_batch["bench_text"].append(eval_texts[idx])
                break
    return new_batch


def diff_strings(string1, string2):
    """Find matching parts between two strings."""
    matcher = difflib.SequenceMatcher(
        None, string1.lower(), string2.lower(), autojunk=False
    )
    matching_blocks = matcher.get_matching_blocks()
    matches = []
    for block in matching_blocks:
        start_a, start_b, length = block
        if length > 5:
            match = string1[start_a : start_a + length]
            matches.append(match)
    return matches


def add_match_stats(example):
    gen_text = " ".join(tokenize(example["completion"]))
    bench_text = " ".join(tokenize(example["bench_text"]))
    matching_parts = diff_strings(gen_text, bench_text)
    match = " ".join("".join(matching_parts).split())
    example["diff"] = matching_parts
    example["diff_ratio"] = len(match) / len(bench_text) if len(bench_text) > 0 else 0
    example["diff_length"] = len(match)
    example["longest_diff_part"] = max(matching_parts, key=len, default="")
    example["longest_diff_part_length"] = len(example["longest_diff_part"])
    return example



def main(args):
    dataset = load_dataset("google/xquad", f"xquad.en")
    indices = []
    L = len(dataset["validation"])
    i = 0
    while i < L:
        indices.append(i)
        example = dataset["validation"][i]
        j = i + 1
        next_example = dataset["validation"][j]
        while j < L and example["context"].strip() == next_example["context"].strip():
            j += 1
            if j < L:
                next_example = dataset["validation"][j]
        i = j
    eval_dataset = dataset["validation"].select(indices).select_columns(["context"])
    print(eval_dataset)
    xquad_eval_ngrams, xquad_eval_datasets, xquad_eval_texts = {}, [], []
    for example in tqdm(eval_dataset):
        tokens = tokenize(example["context"])
        ngrams = get_ngrams(tokens, 10)
        if ngrams:
            idx = len(xquad_eval_texts)
            xquad_eval_ngrams.update(zip(ngrams, [idx] * len(ngrams)))
            xquad_eval_datasets.append(example.get("task_name", "xquad"))
            xquad_eval_texts.append(example["context"])

    LANGUAGES =  [
        "Basque",
        "Hausa",
        "Igbo",
        "Kinyarwanda",
        "Nepali",
        "Somali",
        "Sundanese",
        "Swahili",
        "Urdu",
        "Xhosa",
    ]

    for language in LANGUAGES:
        print(f"{'-'*100}\n{language}\n{'-'*100}")
        list_of_eval_datasets = []
        for evaluation_dataset in ["flores", "ntrex", "tico"]:
            try:
                eval_data = get_datasets(evaluation_dataset, language)
                eval_data = eval_data.select_columns(["sentence"])
                list_of_eval_datasets.extend([eval_data["devtest"], eval_data["dev"]])
            except Exception as e:
                print(f"Exception: {e}")
                continue
        eval_data = concatenate_datasets(list_of_eval_datasets)
        print(eval_data)
        # Load the evaluation data to build n-grams index
        eval_ngrams, eval_datasets, eval_texts = {}, [], []
        for example in tqdm(eval_data):
            tokens = tokenize(example["sentence"])
            ngrams = get_ngrams(tokens, 10)
            if ngrams:
                idx = len(eval_texts)
                eval_ngrams.update(zip(ngrams, [idx] * len(ngrams)))
                eval_datasets.append(example.get("task_name", "unknown"))
                eval_texts.append(example["sentence"])

        list_of_paths = [
            os.path.join(PATH, "data/less/gemma-3-27b-it/topxgen/T=0.0"),
            os.path.join(PATH, "data/less/gemma-3-27b-it/topxgen/T=0.5"),
            os.path.join(PATH, "data/less/gemma-3-27b-it/topxgen/T=1.0"),
            os.path.join(PATH, "data/wiki/gemma-3-27b-it/topxgen/T=0.0"),
            os.path.join(PATH, "data/wiki/gemma-3-27b-it/topxgen/T=0.5"),
            os.path.join(PATH, "data/wiki/gemma-3-27b-it/topxgen/T=1.0"),
            os.path.join(PATH, "data/wiki/gemma-3-27b-it/topxgen/T=1.2"),
            os.path.join(PATH, "data/wiki/gemma-3-27b-it/topxgen/T=2.0"),
            os.path.join(PATH, "data/wiki/gemma-3-27b-it/self-instruct/T=1.0"),
            os.path.join(PATH, "data/wiki/gemma-3-27b-it/knn-instruct/T=1.0"),
            os.path.join(PATH, "data/less/gpt-4o-mini-2024-07-18/self-instruct/T=1.0"),
            os.path.join(PATH, "data/less/gpt-4o-mini-2024-07-18/topxgen/T=1.0"),
            os.path.join(PATH, "data/less/gpt-4o-mini-2024-07-18/topxgen/T=0.0"),
            os.path.join(PATH, "data/wiki/gpt-4o-mini-2024-07-18/topxgen/T=1.0"),
        ]

        for path in list_of_paths:
            if not os.path.exists(path) or not os.path.exists(
                os.path.join(path, f"{language}.jsonl")
            ):
                continue
            sentences = []
            with open(os.path.join(path, f"{language}.jsonl"), "r") as fin:
                for line in fin:
                    sentences.append(json.loads(line)["text"])
            train_data = Dataset.from_dict({"completion": sentences})
            A = (
                f"{'-'*100}\n"
                f"path: {path}\n"
                f"{train_data}\n"
                f"{'-'*100}"
            )
            print(A)
            contamination_report = train_data.map(
                lambda batch: retrieve_ngrams_batch(
                    batch, eval_ngrams, eval_datasets, eval_texts, 10
                ),
                batched=True,
                batch_size=1000,
                num_proc=16,
                remove_columns=train_data.column_names,
            )

            contamination_report = contamination_report.map(
                lambda example: add_match_stats(example), num_proc=16
            )

            for j, example in enumerate(contamination_report):
                print(f"{j+1}. {example['completion']}\n{example['bench_text']}")

            first = [example["completion"] for example in contamination_report]
            for filename in os.listdir(path):
                # Self-improvement
                # if filename != "<your_bt_filename>":
                #   continue
                if os.path.isdir(os.path.join(path, filename)) and os.path.exists(
                    os.path.join(path, filename, f"{language}_translate.jsonl")
                ):
                    sentences = []
                    with open(
                        os.path.join(path, filename, f"{language}_translate.jsonl"), "r"
                    ) as fin:
                        for line in fin:
                            sentences.extend(json.loads(line)["translations"])
                        train_data = Dataset.from_dict({"completion": sentences})

                    train_data = Dataset.from_dict({"completion": sentences})
                    contamination_report = train_data.map(
                        lambda batch: retrieve_ngrams_batch(
                            batch,
                            xquad_eval_ngrams,
                            xquad_eval_datasets,
                            xquad_eval_texts,
                            10,
                        ),
                        batched=True,
                        batch_size=1000,
                        num_proc=16,
                        remove_columns=train_data.column_names,
                    )

                    contamination_report = contamination_report.map(
                        lambda example: add_match_stats(example), num_proc=16
                    )
                    print(
                        f"{'='*100}\n"
                        f"path: {os.path.join(path, filename)}/{language}_translate.jsonl\n"
                        f"{train_data}\n"
                        f"{'='*100}"
                    )
                    for j, example in enumerate(contamination_report):
                        print(
                            f"{j+1}. {example['completion']}\n{example['bench_text']}"
                        )
                    second = [example["completion"] for example in contamination_report]

                    # """
                    input_path = os.path.join(
                        path, filename, f"{language}_translate.jsonl"
                    )
                    output_path = os.path.join(
                        path, filename, f"{language}_translate_filtered.jsonl"
                    )
                    skip = 0
                    skip_v2 = 0
                    with open(input_path, "r", encoding="utf-8") as infile, open(
                        output_path, "w", encoding="utf-8"
                    ) as outfile:
                        for line in infile:
                            dico = json.loads(line)
                            if dico["sentence"] in first:
                                skip += 1
                                print(f"SKIP {skip}")
                                continue
                            # English
                            translations = dico["translations"]
                            # Target
                            propositions = dico["propositions"]
                            indices = [
                                i
                                for i, translation in enumerate(translations)
                                if translation not in second
                            ]
                            if len(translations) != len(indices):
                                skip_v2 += 1
                                print(f"SKIP V2 {skip_v2}; {len(translations)} != {len(indices)}")
                            obj = dico.copy()
                            obj["translations"] = [translations[i] for i in indices]
                            obj["propositions"] = [propositions[i] for i in indices]
                            obj["scores"] = [dico["scores"][i] for i in indices]

                            outfile.write(json.dumps(obj) + "\n")
                    # """
            # print(f"{'-'*100}\nAFTER\n{'-'*100}\n{contamination_report}\n{'-'*100}")


if __name__ == "__main__":
   args = parse_args()
   main(args)