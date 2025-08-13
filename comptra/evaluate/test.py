import multiprocess as mp
from tqdm import tqdm
import numpy as np
import itertools
import json
import os
from comptra.data.dataset import get_datasets
from metricx23.models import MT5ForRegression
import torch
import transformers
from datasets import Dataset
import time

from comet import load_from_checkpoint, download_model
from sacrebleu.metrics import BLEU, CHRF
from scipy import stats
import argparse

bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)

CROP_LENGTH = 300

from comptra.utils import _stop_at_stop_token

STOP_WORDS = [
    "###",
    "\n" * 5,
    "\n\n---",
    "://:",
    "://",
    "____",
    "....",
    ". . . .",
    "strong>strong>",
    "Q:",
    "\nProblem:",
    "://",
    "\nA:",
    "<|eot_id|>",
    "<|start_header_id|>",
    "\n\nFinal Answer:",
    "\n\nProblem:",
    "\n\nInput:",
    "#include",
    "[INST]",
    "\nHuman:",
    "\nNote:",
    "<end_of_turn>",
    "<EOS_TOKEN>",
    "using System;",
    "\ufeffusing System;",
    "2019-2020 Undergraduate Catalog",
    "2020-2021 Undergraduate Catalog",
    "2021-2022 Undergraduate Catalog",
    "2019-2020 Undergraduate Bulletin",
    "2020-2021 Undergraduate Bulletin",
    "2021-2022 Undergraduate Bulletin",
    "2020-04-20 Best 2005 Chevy Silverado",
    "# 1999–2000 in Scottish football",
    "1_0_0_0_0_0_",
    "10000000000000000",
    "Home \u00bb News \u00bb 20",
    "Home \u00bb Blog",
    "Home \u00bb ",
    "2020-04-20 ",
    "\ufeff\ufeff\ufeff",
    "Home » Blog",
    "Home » News",
    "# 2019–20 Liga 1",
    "# 2019–20 Liga I",
    "# 2016–17 Liga 1",
    "a.a. .. ... ..",
    '="">',
    ".....",
    "\n\n\n:",
    ">>\n>",
    "```\>",
    "````",
    '="true">',
    "....]",
    "\n>>\n",
    "=\>=",
    "\n```\n",
    "\n\n\n,",
    "\n\n\n`",
    "a\naa\na",
    "a\na\na\na",
    ",','",
    ">\n>\n>\n>",
    '="text"',
    "<h3>",
    "<h1>",
    "\*\*u  ",
    "*\n*u\n*u",
    "।\n।\n।",
    "a,\n\n,",
    ">>>>",
    "\n.\n.\n.",
    "# 1999–2000 Liga I",
    "# 2019–2020-",
    "# 2022–23 Liga",
    "1990s, 2000s, 2010s, 2020s",
    "1980s, 1990s, 2000s,",
    "1960s, 1970s, 1980s",
    "1970s, 1980s, 1990s",
    "1980's, 1990's, 2000's, 2010's",
    "2010s, 2010s,",
    "Home / News / Sports News ",
    "import React,",
    " <?php",
    "# 2022\u201323 Liga 1 U-21",
    "A 2019–2020-as",
    "# 1999\u20132000 Serie A",
    "000,000,000,000",
    "Home / News / ",
    "Home > News >",
    "2021-2022 Undergraduate and Graduate Academic",
    "# 2019\u20132020-as",
    "# 1999\u20132000 Serie B",
    "1916: The Easter Rising",
    "1968: The Year That",
    "10 9 8 7 6 5 4 3 2 1",
    "1 2 3 4 5 6 7 8 9 10",
    "1 3 5 7 9",
    "# 2020年",
    "# 2016年",
    "# 1988年",
    "# 1976年",
    "## Referências",
    "1999-2000, 2000-2001, 2001-2002",
    "#2016-01-01 00:00:00",
    "#2019-01-03 00:00:00",
    "# 1999\u20132000 in English football",
    "# 1980\u5e74\u30e2\u30b9",
    " 1.1.1.1.1.1.1",
    "import { Component, OnInit }",
    "10.1007/978",
    "6:30, 7:30, 8:30",
    "1999-2000, 2000-2001,",
    "# 2008\u5e74\u590f",
    "# 1999\u20132000 Cypriot First",
    "1986–87 Cypriot First ",
    "Tags: c#, asp.net,",
    "# 1996\u5e74\u590f\u5b63",
    "# 1996\u5e74",
    "import { NgModule }",
    "# 1999–2000 AFC",
    "#101 Post by ",
    "#101 -",
    "#102 -",
    "#101: ",
    "#100DaysOfCode",
    "#1000BlackGirlBooks",
    "#MeTo",
    "#BlackLivesMatter",
    "#201",
    "#20",
    "#10",
    " # ",
    "# 2019\u201320 Serie B",
    "# 2009\u201310 Serie A",
    "# 2019\u201320 Serie A",
    "#101 Posted :",
    "#1 Posted",
    "# 1999\u20132000",
    "# 1996–97 ABA",
    "# 1996",
    "## Refer\u00eancias",
    "Home \u203a News",
    "Pretty unique just like her",
    "The following is a list of the books in the series.",
    "The following is a list of the most common problems that arise in the process of writing a dissertation",
    "The author wishes to thank",
    "2021-2022 Catalog >",
    "2020-2021 Catalog >",
    "2019-2020 Catalog >",
    "2018-2019 Catalog >",
    "package com.example.android.miwok;",
    "2019-2020 Academic Catalog >",
    " ﻿",
    "2021-02-23",
    "2021-02-01",
    "2021-02-18",
    "2022-2023 School Year Calendar",
    "**Note:**",
    "**Please",
    "*Please note",
    "**Explanation:**",
    "*This is a rough translation",
    "**\n\n**",
    "'a'a'a'a'",
    "user222222",
    "userControlEvents",
    "_REF",
    "_REFLECTED",
    "_REFLECTIVE",
    "_REFERENCE",
    "2021-02-19 12:00:00",
    "news today 2021-09-23",
    "2021-09-23 11:00:00",
    "nigeria_states.jpg",
    "nigeria_states.kmz",
    "nigeria news today",
    "north korea Archives - Page 2 of 2",
    '"””””',
    "user\nCan you provide a ",
    "user\nCould you please refine ",
    "usercimal",
    "22222222222",
    "2022-2023 Undergraduate and Diploma",
    "2021-02-17 10:00:00",
    "1. A person",
    "nnnnnnnnnnnnnnnnnn",
    "##Input:",
    "ଏହା କରିପାରିବେ ନାହିଁ.",
    "';';'",
    #"1. ",
    "1999-2000:",
    "north_korea.jpg",
    "north-korea-missile",
    "nba\u6bd4\u5206",
    "2021-02-16 06:00:00",
    "i-i-i-i-i",
    "2021-02-1",
    "1960s, 1970s, 2000s,",
    "2021-2022 Undergraduate and Graduate Catalog >",
    "\u0199a\u0199a\u0199a\u0199a\u0199a\u0199a",
    "2020-02-20 Best 2005",
    "2022-2023 School Year Registration",
    "Home/News/World/",
    "2020-02-20 Best ",
    "2019-04-21",
    "The 2019-2020 school year",
    "1990s, 1990s,",
    "2020-02-19 ",
    "2021-2022 School Year Registration",
    "2019-04-10",
    "The 2018-2019 school year"
]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, help="Path to the directory containing the predictions."
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="flores",
        help="Name of the dataset on which you evaluate e.g. flores, ntrex",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Name or path of the evaluation model.",
    )
    parser.add_argument(
        "--number_of_predictions",
        type=int,
        default=100,
        help="Number of predictions to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument(
        "--max_input_length",
        type=int,
        default=1024,
        help="The maximum allowable input sequence length.",
    )
    parser.add_argument(
        "--is_qe", action="store_true", help="Whether to use a QE metric."
    )
    parser.add_argument(
        "--seed", type=int, default=122, help="Seed for random number generation."
    )
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--metric", type=str, help="metric name")
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Languages to evaluate. A space-separated list of capitalized language names.",
    )
    parser.add_argument(
        "--names",
        nargs="+",
        help="Models whose generations we want to evaluate. A space separated-list of names (as reported in comptra/models.py, after the slash '/') e.g. gemma-2-9b-it",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        help="Strategies to evaluate (bm25, SBYS) etc. e.g. bm25s SBYS SONAR",
    )
    parser.add_argument(
        "--alternative",
        type=str,
        default="two-sided",
        help="Alternative for the statistical test. Par default it is 'two-sided' (Alternative hypothesis: the means of the distributions underlying the samples are unequal.)"
    )
    parser.add_argument(
        "--do_not_test",
        action="store_true",
        help="No statistical test."
    )
    return parser.parse_args()

from comptra.utils import is_lang
from comptra.languages import MAPPING_LANG_TO_KEY
from sonar.models.blaser.loader import load_blaser_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
blaser_qe = load_blaser_model("blaser_2_0_qe", device=device).eval()
text_embedder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder", device=device
)

def get_blaser_score(x, y, src, tgt):
    src_embs = text_embedder.predict([x], source_lang=MAPPING_LANG_TO_KEY[src])
    ref_embs = text_embedder.predict([y], source_lang=MAPPING_LANG_TO_KEY[tgt])
    blaser_score = blaser_qe(src=src_embs, mt=ref_embs).item()
    return blaser_score

def main(args):
    languages = args.languages
    names = args.names if args.names else [
        "Meta-Llama-3.1-70B-Instruct-AWQ-INT4",
        "gemma-2-27b-it",
        "command-r-plus-08-2024",
        "Meta-Llama-3.1-8B-Instruct",
        "gemma-2-9b-it",
        "command-r-08-2024",
        "gemma-2-2b-it",
    ]
    strategies = args.strategies if args.strategies else [
        "bm25s",
        "SONAR",
        "SBYS",
        "TEaR",
        "MAPS",
        "ZS",
        "ZSREFINE",
        "REFINE",
        "ZSCoT",
        "FSCoT",
        "NLLB"
        "LCS"
        # "ABLATIONS/A", "ABLATIONS/B", "ABLATIONS/C", "ABLATIONS/D", "ABLATIONS/E"
    ]
    path = args.data_dir
    print(f"PATH: {path}")
    ds_dict = {}

    is_qe = args.is_qe or "qe" in args.model_name_or_path.lower()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_device_batch_size = args.batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_device_batch_size = args.batch_size

    if args.metric == "comet":
        model_path = download_model(args.model_name_or_path)
        print(f"COMET: {args.model_name_or_path, model_path}")
        model = load_from_checkpoint(model_path)
        print(model)
    elif args.metric == "metricx":
        tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-xl")
        print(f"MetricX23: {args.model_name_or_path}")
        model = MT5ForRegression.from_pretrained(
            args.model_name_or_path, torch_dtype=torch.bfloat16
        )
        model.to(device)
        model.eval()

        output_dir = os.path.join(args.data_dir, ".")
        os.makedirs(output_dir, exist_ok=True)

        training_args = transformers.TrainingArguments(
            output_dir=output_dir,
            per_device_eval_batch_size=per_device_batch_size,
            dataloader_pin_memory=False,
        )
        trainer = transformers.Trainer(model=model, args=training_args,)

        # Datasets utilities
        def _make_input(example):
            if is_qe:
                example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " source: "
                    + example["source"]
                )
            else:
                example["input"] = (
                    "candidate: "
                    + example["hypothesis"]
                    + " reference: "
                    + example["reference"]
                )
            return example

        def _tokenize(example):
            return tokenizer(
                example["input"],
                max_length=args.max_input_length,
                truncation=True,
                padding=False,
            )

        def _remove_eos(example):
            example["input_ids"] = example["input_ids"][:-1]
            example["attention_mask"] = example["attention_mask"][:-1]
            return example

    # Statistical significance parameters
    rng = np.random.default_rng(122)
    n_bootstrap = 300
    sample_test_set_size = 500
    test_set_size = args.number_of_predictions  # len(sources)
    bootstrap_indices = [
        list(rng.choice(test_set_size, size=sample_test_set_size, replace=True))
        for _ in range(n_bootstrap)
    ]
    print("Start")
    for name in names:
        print(f"\n-----\nModel name: {name.upper()}\n-----\n")
        store = {}  # {"language": {"strategy": [scores]}}
        for strategy in strategies:
            data_dir = os.path.join(path, f"{strategy}/{name}")
            print(f"data_dir: {data_dir.upper()}")
            if not os.path.exists(data_dir):
                print(f"{data_dir} does not seem to exist.")
                continue
            count = 0
            for filename in os.listdir(data_dir):
                wrong_format = 0
                if filename.endswith("jsonl"):
                    continue
                if filename.count("_") < 2:
                    continue
                features = filename.split("_")
                source = features[0]
                target = features[2]
                if languages and target not in languages:
                    continue
                if source not in ds_dict:
                    ds_src = get_datasets(args.dataset_name_or_path, source)
                    ds_dict[source] = ds_src
                else:
                    ds_src = ds_dict[source]

                if target not in ds_dict:
                    ds_tgt = get_datasets(args.dataset_name_or_path, target)
                    ds_dict[target] = ds_tgt
                else:
                    ds_tgt = ds_dict[target]

                print(f"source = {source}, target = {target}")

                sources = [
                    example["sentence"] for example in ds_dict[source]["devtest"]
                ][0 : args.number_of_predictions]
                targets = [
                    example["sentence"] for example in ds_dict[target]["devtest"]
                ][0 : args.number_of_predictions]
                try:
                    predictions = []
                    local = os.path.join(data_dir, filename)
                    with open(os.path.join(local, "translate_0.jsonl"), "r") as fin:
                        for j, line in enumerate(fin):
                            prediction = json.loads(line)["translation"]
                            trigger_1 = "sentence is thus:"
                            if trigger_1 in prediction:
                                prediction = prediction[
                                    prediction.find(trigger_1) + len(trigger_1) :
                                ]
                            else:
                                trigger = (
                                    "II. Final translation"
                                    if "II. Final translation" in prediction
                                    else "III. Final translation"
                                )
                                if trigger in prediction:
                                    prediction = prediction[
                                        prediction.find(trigger) + len(trigger) :
                                    ]
                            if len(prediction) > 1000:
                                # Wrong format or infinite repetition
                                wrong_format += 1
                                print(f"Wrong format {wrong_format}: {j+1}")
                                prediction = prediction[:CROP_LENGTH]
                            prediction = _stop_at_stop_token(prediction, STOP_WORDS)
                            predictions.append(prediction)
                except Exception as e:
                    print(f"An error occurred: {e}")
                if len(predictions) != args.number_of_predictions:
                    if len(predictions) < args.number_of_predictions:
                        continue
                    else:
                        predictions = predictions[: args.number_of_predictions]
                if os.path.exists(os.path.join(local, "translate_1.jsonl")):
                    A, B = [], []
                    with open(f"{local}/translate_1.jsonl", "r") as fin:
                        for line in fin:
                            A.append(json.loads(line)["sentence"])
                            B.append(json.loads(line)["translation"])

                if target in store:
                    pass
                else:
                    store[target] = {}
                # Store the prediction per language & strategy
                if "None_1_0" in filename:
                    strategy_key = f"{strategy} + CompTra"
                else:
                    strategy_key = strategy

                if strategy_key in store[target]:
                    pass
                else:
                    store[target][strategy_key] = {}
                k = int(features[3])
                if k in store[target][strategy_key]:
                    pass
                else:
                    store[target][strategy_key][k] = {}

                store[target][strategy_key][k]["predictions"] = predictions

                if "metric" in args.metric:
                    if is_qe:
                        ds = Dataset.from_dict(
                            {"source": sources, "hypothesis": predictions}
                        )
                    else:
                        ds = Dataset.from_dict(
                            {
                                "source": sources,
                                "hypothesis": predictions,
                                "reference": targets,
                            }
                        )
                    ds = ds.map(_make_input)
                    ds = ds.map(_tokenize)
                    ds = ds.map(_remove_eos)
                    ds.set_format(
                        type="torch",
                        columns=["input_ids", "attention_mask"],
                        device=device,
                        output_all_columns=True,
                    )
                    score_predictions, _, _ = trainer.predict(test_dataset=ds)
                    store[target][strategy_key][k]["scores"] = score_predictions

                elif "comet" in args.metric:
                    if is_qe:
                        data = [
                            {"src": sources[i], "mt": predictions[i],}
                            for i in range(len(predictions))
                        ]
                    else:
                        data = [
                            {"src": sources[i], "mt": predictions[i], "ref": targets[i]}
                            for i in range(len(predictions))
                        ]
                    model_output = model.predict(data)
                    store[target][strategy_key][k]["scores"] = model_output.scores

                count += 1
                b = bleu.corpus_score(predictions, [targets]).score
                c = chrf.corpus_score(predictions, [targets]).score
                score = np.mean(
                    [float(pred) for pred in store[target][strategy_key][k]["scores"]]
                )
                print(
                    f"{count}: {filename}\nBLEU = {b}\nchrF++ = {c}\n{args.metric} = {score}\n"
                )

        # Assuming that we translate out of english
        if "English" not in ds_dict:
            ds_dict["English"] = get_datasets(args.dataset_name_or_path, "English")
        
        for language in store:
            candidate_keys = list(store[language].keys())
            for strat in candidate_keys:
                if f"{strat} + CompTra" not in store[language]:
                    continue
                left = store[language][strat]
                right = store[language][f"{strat} + CompTra"]
                store[language][f"Ensemble [{strat} ^ {strat} + CompTra]"] = {}
                for k in left:
                    if k not in right:
                        continue 
                    # initialization
                    store[language][f"Ensemble [{strat} ^ {strat} + CompTra]"][k] = {}
                    # Ensembling per se
                    p_1 = left[k]["predictions"]
                    s_1 = left[k]["scores"]
                    p_2 = right[k]["predictions"]
                    s_2 = right[k]["scores"]
                    sources = [
                        example["sentence"] for example in ds_dict["English"]["devtest"]
                    ][0 : args.number_of_predictions]
                    targets = [
                        example["sentence"] for example in ds_dict[language]["devtest"]
                    ][0 : args.number_of_predictions]
                    p = []
                    s = []
                    for j, (pred_1, pred_2) in tqdm(enumerate(zip(p_1, p_2))):
                        candidates = [pred_1, pred_2]
                        candidate_scores = [s_1[j], s_2[j]]
                        L = [
                            is_lang(candidate, language)
                            * get_blaser_score(
                                x=sources[j], y=candidate, src="English", tgt=language
                            )
                            for candidate in candidates
                        ]
                        p.append(candidates[np.argmax(L)])
                        s.append(candidate_scores[np.argmax(L)])
                    store[language][f"Ensemble [{strat} ^ {strat} + CompTra]"][k]["predictions"] = p
                    store[language][f"Ensemble [{strat} ^ {strat} + CompTra]"][k]["scores"] = s

        for language in store:
            print(f"strategies: {list(store[language].keys())}")

            targets = [example["sentence"] for example in ds_dict[language]["devtest"]]
            print(f"{language}: {targets[0]}")
            if args.do_not_test:
                continue
                
            for a, b in itertools.combinations(list(store[language].keys()), 2):
                # Consider every single pair
                for k in store[language][a]:
                    for k2 in store[language][b]:
                        # if k != k2: continue
                        start = time.time()
                        predictions_1 = store[language][a][k]["predictions"]
                        predictions_2 = store[language][b][k2]["predictions"]

                        scores_1 = store[language][a][k]["scores"]
                        scores_2 = store[language][b][k2]["scores"]

                        bootstrap_scores_1 = [
                            np.mean([scores_1[j] for j in indices])
                            for indices in bootstrap_indices
                        ]

                        bootstrap_scores_2 = [
                            np.mean([scores_2[j] for j in indices])
                            for indices in bootstrap_indices
                        ]

                        p = mp.Pool(args.num_workers)

                        def f(indices):
                            b_ = bleu.corpus_score(
                                [predictions_1[p] for p in indices],
                                [[targets[p] for p in indices]],
                            ).score
                            return b_

                        bootstrap_bleu_scores_1 = p.map(f, bootstrap_indices)

                        def f(indices):
                            c_ = chrf.corpus_score(
                                [predictions_1[p] for p in indices],
                                [[targets[p] for p in indices]],
                            ).score
                            return c_

                        bootstrap_chrf_scores_1 = p.map(f, bootstrap_indices)

                        def f(indices):
                            b_ = bleu.corpus_score(
                                [predictions_2[p] for p in indices],
                                [[targets[p] for p in indices]],
                            ).score
                            return b_

                        bootstrap_bleu_scores_2 = p.map(f, bootstrap_indices)

                        def f(indices):
                            c_ = chrf.corpus_score(
                                [predictions_2[p] for p in indices],
                                [[targets[p] for p in indices]],
                            ).score
                            return c_

                        bootstrap_chrf_scores_2 = p.map(f, bootstrap_indices)

                        alternative = args.alternative  # greater

                        score_output = stats.ttest_rel(
                            bootstrap_scores_1,
                            bootstrap_scores_2,
                            alternative=alternative,
                        )

                        bleu_output = stats.ttest_rel(
                            bootstrap_bleu_scores_1,
                            bootstrap_bleu_scores_2,
                            alternative=alternative,
                        )

                        chrf_output = stats.ttest_rel(
                            bootstrap_chrf_scores_1,
                            bootstrap_chrf_scores_2,
                            alternative=alternative,
                        )
                        duration = round(time.time() - start, 2)
                        pvalue = score_output.pvalue
                        bleu_pvalue = bleu_output.pvalue
                        chrf_pvalue = chrf_output.pvalue

                        # If two-sided, p-value < threshold => reject the null hypothesis of identical average scores
                        print(
                            f"===\nModel: {name}\n{a} in {k}-shot vs {b} in {k2}-shot for the {language} language.\nscore={score_output}\nDuration: {duration} seconds\nBLEU={bleu_output}\nchrF++={chrf_output}"
                        )

                        def get_message(x):
                            if x < 0.001:
                                return f"NOT EQUAL i.e is SIGNIFICANT at 0.001"
                            elif x < 0.05:
                                return f"NOT EQUAL i.e is SIGNIFICANT at 0.05"
                            if x >= 0.05:
                                return "is NOT SIGNIFICANT at 0.05"
                            if x >= 0.001:
                                return "is NOT SIGNIFICANT at 0.001"
                            return None

                        p1, p2, p3 = (
                            get_message(pvalue),
                            get_message(bleu_pvalue),
                            get_message(chrf_pvalue),
                        )
                        print(
                            f"\n1. {args.metric} -> {p1}\n2. BLEU -> {p2}\n3. chrF++ -> {p3}\n===\n"
                        )
    print("END!")


if __name__ == "__main__":
    args = parse_args()
    main(args)
