# coding=utf-8
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs inference with a MetricX model."""

import dataclasses
import json
import os

#from comptra.evaluate.metricx24.models import MT5ForRegression
from models import MT5ForRegression
import torch
import transformers
from datasets import Dataset

@dataclasses.dataclass
class Arguments:
    """Prediction command-line arguments."""

    tokenizer: str = dataclasses.field(
        metadata={"help": "The name of the tokenizer"},
    )

    model_name_or_path: str = dataclasses.field(
        metadata={
            "help": (
                "Path to pretrained model or model identifier from"
                " huggingface.co/models"
            )
        }
    )
    dataset_name_or_path: str = dataclasses.field(
        metadata = {
            "help": "Name of the dataset on which you evaluate e.g. flores, ntrex"
        }
    )

    max_input_length: int = dataclasses.field(
        metadata={"help": "The maximum allowable input sequence length."},
    )

    batch_size: int = dataclasses.field(
        metadata={"help": "The global prediction batch size."},
    )

    input_dir: str = dataclasses.field(metadata={"help": "The folder containing the hypothesis files."})
    
    number_of_predictions: int = dataclasses.field(metadata={"help": "Number of predictions to evaluate."})

    qe: bool = dataclasses.field(
        metadata={"help": "Indicates the metric is a QE metric."},
        default=False,
    )
    seed: int = dataclasses.field(
        metadata={"help": "Seed for random number generation."},
        default=122
    )
    ensembling: bool = dataclasses.field(
        metadata={"help": "Whether to compute the ensembling scores"},
        default=False
    )

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

from sacrebleu.metrics import BLEU, CHRF
bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)

from comptra.data.dataset import get_datasets
import numpy as np

CROP_LENGTH = 300

def main() -> None:
    parser = transformers.HfArgumentParser(Arguments)
    (args,) = parser.parse_args_into_dataclasses()

    # Dictionary of datasets
    ds_dict = {}
    rng = np.random.default_rng(args.seed)

    is_qe = args.qe or "QE" in args.model_name_or_path.upper()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_device_batch_size = args.batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_device_batch_size = args.batch_size

    tokenizer = transformers.AutoTokenizer.from_pretrained(args.tokenizer)
    print(f"MetricX-24: {args.model_name_or_path}")
    model = MT5ForRegression.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    output_dir = os.path.join(args.input_dir, ".") 
    os.makedirs(output_dir, exist_ok=True)
    
    training_args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_pin_memory=False,
        report_to="none"
    )
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
    )

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
    
    count = 0
    dico = {}
    for filename in os.listdir(args.input_dir):
        wrong_format = 0
        if filename.endswith("jsonl"):
            continue
        if filename.count("_") < 2:
            continue
        features = filename.split("_")
        source = features[0]
        target = features[2]
        if args.dataset_name_or_path in ['tico', 'ood'] and target not in ["Amharic", "Khmer", "Lingala", "Luganda", "Tamil"]:
            continue
        #if args.dataset_name_or_path in ["flores"] and target not in ["Amharic", "Burmese", "Fijian", "Khmer", "Lao", "Samoan", "Sinhala", "Tsonga", "Turkmen", "Uyghur", "N\'ko"]:
        #    continue
        if args.dataset_name_or_path in ["ntrex"] and target not in ["Amharic", "Khmer", "Shona", "Somali", "Tswana"]:
            continue
        features = filename.split("_")
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

        sources = [example["sentence"] for example in ds_src["devtest"]][
            0 : args.number_of_predictions
        ]
        targets = [example["sentence"] for example in ds_tgt["devtest"]][
            0 : args.number_of_predictions
        ]
        try:
            predictions = []
            local = os.path.join(args.input_dir, filename)
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
                    predictions.append(prediction)
        except Exception as e:
            print(f"An error occurred: {e}")
        if len(predictions) != args.number_of_predictions:
            if len(predictions) < args.number_of_predictions:
                continue
            else:
                predictions = predictions[: args.number_of_predictions]
        if is_qe:
            ds = Dataset.from_dict(
                {
                    "source": sources,
                    "hypothesis": predictions
                }
            )
        else:
            ds = Dataset.from_dict(
                {
                    "source": sources,
                    "hypothesis": predictions,
                    "reference": targets
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
         
        if os.path.exists(os.path.join(local, "translate_1.jsonl")) and is_qe:
            A, B = [], []
            with open(f"{local}/translate_1.jsonl", "r") as fin:
                for line in fin:
                    A.append(json.loads(line)["sentence"])
                    B.append(json.loads(line)["translation"])
            ds_intermediate = Dataset.from_dict({"source": A, "hypothesis": B})
            ds_intermediate = ds_intermediate.map(_make_input)
            ds_intermediate = ds_intermediate.map(_tokenize)
            ds_intermediate = ds_intermediate.map(_remove_eos)
            ds_intermediate.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
                device=device,
                output_all_columns=True,
            )
            score_predictions, _, _ = trainer.predict(test_dataset=ds_intermediate)
            print(f"Intermediate score: {np.mean([float(pred) for pred in score_predictions])}")

        score_predictions, _, _ = trainer.predict(test_dataset=ds)  
        count += 1
        b = bleu.corpus_score(predictions, [targets]).score
        c = chrf.corpus_score(predictions, [targets]).score
        score = np.mean([float(pred) for pred in score_predictions])
        print(f"{count}: {filename}\nBLEU = {b}\nchrF++ = {c}\nMetricX23 = {score}\n")

        key = "_".join(filename.split("_")[:-3]) + "_"
        # print(f"key: {key}")
        _, depth, refine = filename[len(key) :].split("_")
        depth, refine = int(depth), int(refine)
        if key in dico:
            dico[key][depth] = predictions
        else:
            dico[key] = {depth: predictions}
    
    count = 0
    for key in dico:
        if len(dico[key]) != 2:
            continue
        output_filename = f"{key}.jsonl"
        os.makedirs(os.path.join(args.input_dir, "scores"), exist_ok=True)
        output_path = os.path.join(
            os.path.join(args.input_dir, "scores"), output_filename
        )
        features = key.split("_")
        source = features[0]
        target = features[2]
        print(f"source = {source}, target = {target}")
        sources = [example["sentence"] for example in ds_dict[source]["devtest"]][
            0 : args.number_of_predictions
        ]
        targets = [example["sentence"] for example in ds_dict[target]["devtest"]][
            0 : args.number_of_predictions
        ]
        # Ensembling
        if args.ensembling:
            try:
                from comptra.utils import is_lang

                values = [v for _, v in dico[key].items()]
                predictions = []
                pred2 = []
                for j, candidates in enumerate(zip(*values)):
                    L = [
                        is_lang(candidate, target)
                        * get_blaser_score(
                            x=sources[j], y=candidate, src=source, tgt=target
                        )
                        for candidate in candidates
                    ]
                    predictions.append(candidates[np.argmax(L)])
                    if is_lang(candidates[0], target):
                        if is_lang(candidates[1], target):
                            pred2.append(candidates[int(rng.choice([0, 1], size=1)[0])])
                        else:
                            pred2.append(candidates[0])
                    else:
                        if is_lang(candidates[1], target):
                            pred2.append(candidates[1])
                        else:
                            pred2.append(candidates[int(rng.choice([0, 1], size=1)[0])])

                b = bleu.corpus_score(predictions, [targets]).score
                b2 = bleu.corpus_score(pred2, [targets]).score

                c = chrf.corpus_score(predictions, [targets]).score
                c2 = chrf.corpus_score(pred2, [targets]).score
                if is_qe:
                    ds = Dataset.from_dict(
                        {
                            "source": sources,
                            "hypothesis": predictions
                        }
                    )
                    ds2 = Dataset.from_dict(
                        {
                            "source": sources,
                            "hypothesis": pred2
                        }
                    )                    
                else:
                    ds = Dataset.from_dict(
                        {
                            "source": sources,
                            "hypothesis": predictions,
                            "reference": targets
                        }
                    )
                    ds2 = Dataset.from_dict(
                        {
                            "source": sources,
                            "hypothesis":pred2,
                            "reference": targets
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

                ds2 = ds2.map(_make_input)
                ds2 = ds2.map(_tokenize)
                ds2 = ds2.map(_remove_eos)
                ds2.set_format(
                    type="torch",
                    columns=["input_ids", "attention_mask"],
                    device=device,
                    output_all_columns=True,
                )

                score_predictions, _, _ = trainer.predict(test_dataset=ds)
                metricx_score = np.mean([float(pred) for pred in score_predictions])

                score_predictions2, _, _ = trainer.predict(test_dataset=ds2)
                metricx_score2 = np.mean([float(pred) for pred in score_predictions2])

                count += 1
                print(
                    f"BLASER {count}: {key}\nBLEU = {b}\nchrF++ = {c}\nMetricX = {metricx_score}\n"
                )
                print(
                    f"Random {count}: {key}\nBLEU = {b2}\nchrF++ = {c2}\nMetricX = {metricx_score2}\n"
                )
            except Exception as e:
                print(f"An error occurred: {e}")

    print("END")

if __name__ == "__main__":
    main()