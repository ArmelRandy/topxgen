import re
from termcolor import colored


def parse_and_check_numerical_answer(answer, min=None, max=None):
    attempt = parse_numerical_answer(answer, min, max)
    if attempt is not None:
        if attempt < min or attempt > max:
            return None
        return attempt

    return None


def parse_numerical_answer(answer, min=None, max=None):
    # get all numbers in a string
    numbers = re.findall(r"\d+", answer)
    if len(numbers) == 1:
        return int(numbers[0])

    # check if the answer is in form ['100'] and extract the number
    r1 = re.match(r"^\[['\"][0-9]*['\"]\]$", answer)
    if r1 is not None:
        return int(answer[2:-2])

    if max is not None:
        # check if the answer is in a form of 0/100
        r2 = re.match(rf"^[0-9]*/{max}$", answer)
        if r2 is not None:
            return int(answer.split("/")[0])

    return None


def validate_number(x, min=0, max=100):
    attempt = parse_and_check_numerical_answer(x, min, max)
    if attempt is not None:
        return attempt
    return None


def parse_classes(answer, classes):
    final_class = None
    for i in range(len(classes)):
        if classes[i].lower() in answer.lower():
            if final_class is None:
                final_class = i
            else:
                print(colored(f"Two classes found in answer {answer}", "red"))
                return None

    return final_class


def validate_stars(x):
    x = x.lower()
    # try to find all possible answers as sometimes it seems to be explaining itself
    possible_answers = set()

    # check if string x contains * characters
    if "*" in x:
        possible_answers.add(x.count("*"))
    if "★" in x:
        possible_answers.add(x.count("★"))

    x = f" {x} ".replace("\n", " ")
    # possible answers: "five stars", "5 stars", "five", "five starts: perfect translation", ...
    if " one " in x or "1 star" in x:
        possible_answers.add(1)
    if " two " in x or "2 star" in x:
        possible_answers.add(2)
    if " three " in x or "3 star" in x:
        possible_answers.add(3)
    if " four " in x or "4 star" in x:
        possible_answers.add(4)
    if " five " in x or "5 star" in x:
        possible_answers.add(5)

    numerical = parse_numerical_answer(x)
    if numerical is not None:
        possible_answers.add(numerical)

    if len(possible_answers) == 1:
        answer = possible_answers.pop()
        if 1 <= answer <= 5:
            return answer
    return None


language_codes = {
    "en": "English",
    "de": "German",
    "zh": "Chinese",
    "ru": "Russian",
}

prompts = {
    "GEMBA-DA": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100, where a score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False,
    },
    "GEMBA-DA_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to human reference on a continuous scale 0 to 100 where score of zero means "no meaning preserved" and score of one hundred means "perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: {reference_seg}\n{target_lang} machine translation: "{target_seg}"\nScore: ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": True,
    },
    "GEMBA-SQM": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} on a continuous scale from 0 to 100 that starts on "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": False,
    },
    "GEMBA-SQM_ref": {
        "prompt": 'Score the following machine translation from {source_lang} to {target_lang} with respect to the human reference on a continuous scale from 0 to 100 that starts with "No meaning preserved", goes through "Some meaning preserved", then "Most meaning preserved and few grammar mistakes", up to "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} machine translation: "{target_seg}"\nScore (0-100): ',
        "validate_answer": lambda x: validate_number(x),
        "use_ref": True,
    },
    "GEMBA-stars": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} translation: "{target_seg}"\nStars: ',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": False,
    },
    "GEMBA-stars_ref": {
        "prompt": 'Score the following translation from {source_lang} to {target_lang} with respect to the human reference with one to five stars. Where one star means "Nonsense/No meaning preserved", two stars mean "Some meaning preserved, but not understandable", three stars mean "Some meaning preserved and understandable", four stars mean "Most meaning preserved with possibly few grammar mistakes", and five stars mean "Perfect meaning and grammar".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} translation: "{target_seg}"\nStars: ',
        "validate_answer": lambda x: validate_stars(x),
        "use_ref": True,
    },
    "GEMBA-classes": {
        "prompt": 'Classify the quality of machine translation from {source_lang} to {target_lang} into one of following classes: "No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation".\n\n{source_lang} source: "{source_seg}"\n{target_lang} machine translation: "{target_seg}"\nClass: ',
        "use_ref": False,
        "validate_answer": lambda x, classes=[
            "No meaning preserved",
            "Some meaning preserved, but not understandable",
            "Some meaning preserved and understandable",
            "Most meaning preserved, minor issues",
            "Perfect translation",
        ]: parse_classes(x, classes),
        "max_tokens": 100,
    },
    "GEMBA-classes_ref": {
        "prompt": 'Classify the quality of machine translation from {source_lang} to {target_lang} with respect to the human reference into one of following classes: "No meaning preserved", "Some meaning preserved, but not understandable", "Some meaning preserved and understandable", "Most meaning preserved, minor issues", "Perfect translation".\n\n{source_lang} source: "{source_seg}"\n{target_lang} human reference: "{reference_seg}"\n{target_lang} machine translation: "{target_seg}"\nClass: ',
        "use_ref": True,
        "validate_answer": lambda x, classes=[
            "No meaning preserved",
            "Some meaning preserved, but not understandable",
            "Some meaning preserved and understandable",
            "Most meaning preserved, minor issues",
            "Perfect translation",
        ]: parse_classes(x, classes),
        "max_tokens": 100,
    },
}

from comptra.data.dataset import get_datasets
from comptra.sampler import OpenAISampler
import numpy as np
import argparse
import json
import os

CROP_LENGTH = 300

from sacrebleu.metrics import BLEU, CHRF

bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--api_key", type=str, help="OPENAI API KEY"
    )
    parser.add_argument(
        "--data_dir", type=str, help="Path to the directory containing the predictions."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature of the generation"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="Nucleus sampling parameter."
    )
    parser.add_argument(
        "--request_batch_size", type=int, default=16, help="Request batch size."
    )
    parser.add_argument(
        "--number_of_predictions",
        type=int,
        default=100,
        help="Number of predictions to evaluate",
    )
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="flores",
        help="Name of the dataset on which you evaluate e.g. flores, ntrex",
    )
    parser.add_argument("--is_qe", action="store_true", help="Whether to use Gemba QE.")
    return parser.parse_args()


ds_dict = {}


def main(args):
    header = "Make sure to directly return the score. Nothing else.\n\n"
    dico = {}
    count = 0
    is_qe = args.is_qe
    model = OpenAISampler(
        api_key=args.api_key if args.api_key else os.environ.get(
            "OPENAI_API_KEY",
            "<YOUR_API_KEY>"
        ),
        model_name_or_path="gpt-4o-mini-2024-07-18",
        tokenizer_name_or_path="gpt-4o-mini-2024-07-18",
        src="English",
        tgt="French",
        template=None,
        merge_prompt="vanilla",
        method_translate="vanilla",
        selection_method="greedy",
        nllb_name_or_path=None
    )

    generation_kwargs = {
        "max_new_tokens": 50,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": 1.0,
        "num_return_sequences": 1,
        "num_beams": 1,
        "do_sample": False,
        "request_batch_size": args.request_batch_size,
        "verbose": True
    }

    for filename in os.listdir(args.data_dir):
        wrong_format = 0
        if filename.endswith("jsonl"):
            continue
        if filename.count("_") < 2:
            continue
        features = filename.split("_")
        source = features[0]
        target = features[2]
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

        sources = [example["sentence"] for example in ds_dict[source]["devtest"]][
            0 : args.number_of_predictions
        ]
        targets = [example["sentence"] for example in ds_dict[target]["devtest"]][
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
        if os.path.exists(os.path.join(local, "translate_1.jsonl")):
            A, B = [], []
            with open(f"{local}/translate_1.jsonl", "r") as fin:
                for line in fin:
                    A.append(json.loads(line)["sentence"])
                    B.append(json.loads(line)["translation"])
            data = [
                header + prompts["GEMBA-DA"]["prompt"].format(
                    source_lang=source,
                    target_lang=target,
                    source_seg=A[i],
                    target_seg=B[i],
                )
                for i in range(len(A))
            ]
            if is_qe:
                model_output = model.generate(prompts=data, **generation_kwargs)
                model_outputs = []
                for element in model_output:
                    model_outputs.append(validate_number(x=element[0]))
                    print(model_outputs[-1])
                gemba_qe_score = round(np.mean(model_outputs), 2)
                print(f"Intermediate score: {gemba_qe_score}")
        key = "_".join(filename.split("_")[:-3]) + "_"
        print(f"key: {key}")
        _, depth, refine = filename[len(key) :].split("_")
        depth, refine = int(depth), int(refine)
        if key in dico:
            dico[key][depth] = predictions
        else:
            dico[key] = {depth: predictions}
        b = bleu.corpus_score(predictions, [targets]).score
        c = chrf.corpus_score(predictions, [targets]).score
        if is_qe:
            data = [
                header + prompts["GEMBA-DA"]["prompt"].format(
                    source_lang=source,
                    target_lang=target,
                    source_seg=sources[i],
                    target_seg=predictions[i],
                )
                for i in range(len(sources))
            ]
        else:
            data = [
                header + prompts["GEMBA-DA_ref"]["prompt"].format(
                    source_lang=source,
                    target_lang=target,
                    source_seg=sources[i],
                    target_seg=predictions[i],
                    reference_seg=targets[i],
                )
                for i in range(len(sources))
            ]
        count += 1
        model_output = model.generate(prompts=data, **generation_kwargs)
        model_outputs = []
        for j, element in enumerate(model_output):
            model_outputs.append(validate_number(x=element[0]))
            print(f"{j+1}. {model_outputs[-1]}")
        gemba_score = round(np.mean(model_outputs), 2)
        print(f"{count}: {filename}\nBLEU = {b}\nchrF++ = {c}\nCOMET = {gemba_score}\n")


if __name__ == "__main__":
    args = parse_args()
    main(args)
