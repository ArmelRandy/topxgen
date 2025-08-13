import os
import json
import argparse
import numpy as np
from datasets import load_dataset
from sacrebleu.metrics import BLEU, CHRF
from comet import load_from_checkpoint, download_model

import torch
from transformers import AutoTokenizer
from load_model_from_huggingface_hub import MTRanker
from comptra.languages import MAPPING_LANG_TO_KEY
from comptra.data.dataset import get_datasets

ds_dict = {}

def get_prompt(
    orig_instruction,
    orig_response_A,
    orig_response_B
):
    return f"Source: {orig_instruction.strip()} Translation 0: {orig_response_A.strip()} Translation 1: {orig_response_B.strip()}"

bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_dir", type=str, help="Path to the directory containing the predictions."
    )
    parser.add_argument(
        "--comet_model_path", type=str, default="Unbabel/wmt22-comet-da"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="ibraheemmoosa/mt-ranker-base"
    )
    parser.add_argument(
        "--number_of_predictions",
        type=int,
        default=100,
        help="Number of predictions to evaluate",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum length for tokenization."
    )
    parser.add_argument("--request_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--verbose", action="store_true", help="Verbose.")
    parser.add_argument(
        "--seed", type=int, default=122, help="Seed for random number generation."
    )
    parser.add_argument(
        "--ensembling", action="store_true", help="Whether to compute the ensembling scores."
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Name of the dataset on which you evaluate e.g. flores, ntrex"
    )
    parser.add_argument("--is_qe", action="store_true", help="specify if the metric is reference-less.")
    return parser.parse_args()

CROP_LENGTH = 300

def main(args):
    dico = {}
    count = 0
    rng = np.random.default_rng(args.seed)
    model_path = download_model(args.comet_model_path)
    model = load_from_checkpoint(model_path)
    is_qe = args.is_qe or "qe" in args.comet_model_path
    for filename in os.listdir(args.data_dir):
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
        if args.dataset_name_or_path in ["flores"] and target not in ["Amharic", "Burmese", "Fijian", "Khmer", "Lao", "Samoan", "Sinhala", "Tsonga", "Turkmen", "Uyghur", "N\'ko"]:
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
            local = os.path.join(args.data_dir, filename)
            with open(os.path.join(local, "translate_0.jsonl"), "r") as fin:
                for j, line in enumerate(fin):
                    prediction = json.loads(line)["translation"]
                    trigger_1 = "sentence is thus:"
                    if trigger_1 in prediction:
                        prediction = prediction[prediction.find(trigger_1) + len(trigger_1): ]
                    else:
                        trigger = "II. Final translation" if "II. Final translation" in prediction else "III. Final translation"
                        if trigger in prediction:
                            prediction = prediction[prediction.find(trigger) + len(trigger): ]
                    if len(prediction) > 1000:
                        # Wrong format or infinite repetition
                        wrong_format += 1
                        print(f"Wrong format {wrong_format}: {j+1}")
                        # print(prediction)
                        prediction = prediction[:CROP_LENGTH]
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
            data = [{"src": A[i], "mt": B[i]} for i in range(len(A))]
            if is_qe:
                model_output = model.predict(data)
                comet_qe_score = model_output.system_score
                print(f"Intermediate score: {comet_qe_score}")
        key = "_".join(filename.split("_")[:-3]) + "_"
        print(f"key: {key}")
        _, depth, refine = filename[len(key):].split("_")
        depth, refine = int(depth), int(refine)
        if key in dico:
            dico[key][depth] = predictions
        else:
            dico[key] = {depth: predictions}
        b = bleu.corpus_score(predictions, [targets]).score
        c = chrf.corpus_score(predictions, [targets]).score
        if "qe" in args.comet_model_path:
            data = [
                {
                    "src": sources[i],
                    "mt": predictions[i],
                }
                for i in range(len(predictions))
            ]
        else:
            data = [
                {"src": sources[i], "mt": predictions[i], "ref": targets[i]}
                for i in range(len(predictions))
            ]
        count += 1
        model_output = model.predict(data)
        comet_score = model_output.system_score
        print(f"{count}: {filename}\nBLEU = {b}\nchrF++ = {c}\nCOMET = {comet_score}\n")
    
    ranker = MTRanker.from_pretrained(args.model_name_or_path)
    tokenizer =  AutoTokenizer.from_pretrained(args.model_name_or_path)

    count = 0
    for key in dico:
        if len(dico[key]) != 2:
            continue
        output_filename = f"{key}.jsonl"
        os.makedirs(os.path.join(args.data_dir, "ranking"), exist_ok=True)
        output_path = os.path.join(
            os.path.join(args.data_dir, "ranking"), output_filename
        )
        features = key.split("_")
        source = features[0]
        target = features[2]
        sources = [example["sentence"] for example in ds_dict[source]["devtest"]][
            0 : args.number_of_predictions
        ]
        targets = [example["sentence"] for example in ds_dict[target]["devtest"]][
            0 : args.number_of_predictions
        ]

        if args.ensembling:
            import fasttext
            from huggingface_hub import hf_hub_download

            model_name_or_path = hf_hub_download(
                repo_id="facebook/fasttext-language-identification", filename="model.bin"
            )
            language_identifier = fasttext.load_model(model_name_or_path)

            def is_lang(sentence, lang):
                """
                Takes as input a sentence and a language and output whether the sentence is written in that language.
                Arguments
                ---------
                    - sentence : str,
                        A given sentence
                    - lang :
                        A language (e.g. English, French, German etc.)
                """
                label, _ = language_identifier.predict(sentence.split("\n")[0])
                label = label[0]
                return MAPPING_LANG_TO_KEY[lang] in label
            
            # Load the quality estimator
            model_path = download_model("Unbabel/wmt20-comet-qe-da")
            qe = load_from_checkpoint(model_path)
        # Ensembling
        if args.ensembling:
            # Easy to parallelize
            # np.array(model_output.scores)
            try:
                # from dac.mt.utils import get_blaser_score, is_lang

                values = [v for _, v in dico[key].items()]
                predictions = []
                pred2 = []
                for j, candidates in enumerate(zip(*values)):
                    """
                    L = [
                        is_lang(candidate, target)
                        * get_blaser_score(
                            x=sources[j], y=candidate, src=source, tgt=target
                        )
                    ]
                    """
                    L = [
                        is_lang(candidate, target) * qe.predict([{"src": sources[j], "mt": candidate}]).system_score
                        for candidate in candidates
                    ]
                    predictions.append(candidates[np.argmax(L)])
                    if is_lang(candidates[0], target):
                        if is_lang(candidates[1], target):
                            pred2.append(candidates[int(rng.choice([0, 1], size=1))])
                        else:
                            pred2.append(candidates[0])
                    else:
                        if is_lang(candidates[1], target):
                            pred2.append(candidates[1])
                        else:
                            pred2.append(candidates[int(rng.choice([0, 1], size=1))])

                b = bleu.corpus_score(predictions, [targets]).score
                b2 = bleu.corpus_score(pred2, [targets]).score

                c = chrf.corpus_score(predictions, [targets]).score
                c2 = chrf.corpus_score(pred2, [targets]).score
                if "qe" not in args.comet_model_path:
                    data = [
                        {"src": sources[i], "mt": predictions[i], "ref": targets[i]}
                        for i in range(len(predictions))
                    ]
                    data2 = [
                        {"src": sources[i], "mt": pred2[i], "ref": targets[i]}
                        for i in range(len(pred2))
                    ]
                else:
                    data = [
                        {"src": sources[i], "mt": predictions[i]}
                        for i in range(len(predictions))
                    ]
                    data2 = [{"src": sources[i], "mt": pred2[i]} for i in range(len(pred2))]
                model_output = model.predict(data)
                comet_score = model_output.system_score
                model_output2 = model.predict(data2)
                comet_score2 = model_output2.system_score

                count += 1
                print(f"BLASER {count}: {key}\nBLEU = {b}\nchrF++ = {c}\nCOMET = {comet_score}\n")
                print(
                    f"Random {count}: {key}\nBLEU = {b2}\nchrF++ = {c2}\nCOMET = {comet_score2}\n"
                )
            except Exception as e:
                print(f"An error occurred: {e}")
        
        # Relative Grading (Pairwise Ranking)
        A = dico[key][0]
        B = dico[key][1]

        prompts = []
        for i in range(len(A)):
            prompt = get_prompt(sources[i], A[i], B[i])
            prompts.append(prompt)
        print(f"===\n{prompts[0]}\n===")
        start = 0
        if os.path.exists(output_path):
            with open(output_path, "r") as fin:
                for line in fin:
                    start += 1
        for i in range(start, len(A), args.request_batch_size):
            inputs = prompts[i : i + args.request_batch_size]
            tokenized_inputs = tokenizer(inputs, padding=True, truncation=True, max_length=args.max_length, return_tensors="pt")
            with torch.no_grad():
                outputs = ranker(**tokenized_inputs)
            predictions = outputs.argmax(-1)
            if args.verbose:
                print("===")
                for j, r in enumerate(predictions):
                    print(f"{j+1} -> {r.item()}")
                print("===")
            answers = []
            for j, prediction in enumerate(predictions):
                if prediction.item() == 0:
                    answer = "A"
                elif prediction.item() == 1:
                    answer = "B"
                else:
                    answer = None
                answers.append(answer)
                print(f"===\n{i + j}. {answer}\n===")
            with open(output_path, "a") as fout:
                for j, answer in enumerate(answers):
                    fout.write(
                        json.dumps(
                            {
                                "source": sources[i + j],
                                "translation_A": A[i + j],
                                "translation_B": B[i + j],
                                "reference": targets[i + j],
                                "score": answer,
                            }
                        )
                        + "\n"
                    )
    print("END")
    
if __name__ == "__main__":
    args = parse_args()
    main(args)