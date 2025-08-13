import os
import json
import argparse
import numpy as np
from sacrebleu.metrics import BLEU, CHRF
from comet import load_from_checkpoint, download_model

from comptra.data.dataset import get_datasets

ds_dict = {}

from comptra.languages import MAPPING_LANG_TO_KEY
from sonar.models.blaser.loader import load_blaser_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
text_embedder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder",
    tokenizer="text_sonar_basic_encoder",
    # device=device,
)


def get_blaser_score(x, y, src, tgt):
    src_embs = text_embedder.predict([x], source_lang=MAPPING_LANG_TO_KEY[src])
    ref_embs = text_embedder.predict([y], source_lang=MAPPING_LANG_TO_KEY[tgt])
    blaser_score = blaser_qe(src=src_embs, mt=ref_embs).item()
    return blaser_score


def get_prompt(
    orig_instruction,
    orig_response_A,
    orig_response_B,
    orig_reference_answer,
    orig_criteria=None,
):
    t = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of two responses strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, choose a better response between Response A and Response B. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (A or B)"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{orig_instruction}

###Response A:
{orig_response_A}

###Response B:
{orig_response_B}

###Reference Answer:
{orig_reference_answer}

###Score Rubric:
{orig_criteria}

###Feedback: """
    return t.format(
        orig_instruction=orig_instruction,
        orig_response_A=orig_response_A,
        orig_response_B=orig_response_B,
        orig_reference_answer=orig_reference_answer,
        orig_criteria=orig_criteria,
    )


bleu = BLEU(tokenize="flores200")
chrf = CHRF(word_order=2)

CROP_LENGTH = 300

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
        "--comet_model_path", type=str, default="Unbabel/wmt22-comet-da"
    )
    parser.add_argument(
        "--model_name_or_path", type=str, default="prometheus-eval/prometheus-7b-v2.0"
    )
    parser.add_argument(
        "--number_of_predictions",
        type=int,
        default=100,
        help="Number of predictions to evaluate",
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature of the generation."
    )
    parser.add_argument(
        "--top_p", type=float, help="Top_p parameter, for nucleus sampling."
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=1,
        help="Number of output sequences to return for the given prompt. Should be less or equal to `num_beams` in case of beam search.",
    )
    parser.add_argument(
        "--num_beams", type=int, default=1, help="Number of beams, for beam search."
    )
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty.")
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=750)
    parser.add_argument("--request_batch_size", type=int, default=8, help="batch size")
    parser.add_argument("--verbose", action="store_true", help="Verbose.")
    parser.add_argument(
        "--seed", type=int, default=122, help="Seed for random number generation."
    )
    parser.add_argument(
        "--ensembling",
        action="store_true",
        help="Whether to compute the ensembling scores.",
    )
    return parser.parse_args()


def main(args):
    dico = {}
    count = 0
    rng = np.random.default_rng(args.seed)
    model_path = download_model(args.comet_model_path)
    model = load_from_checkpoint(model_path)
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
            data = [{"src": A[i], "mt": B[i]} for i in range(len(A))]
            if "qe" in args.comet_model_path:
                model_output = model.predict(data)
                comet_qe_score = model_output.system_score
                print(f"Intermediate score: {comet_qe_score}")
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
        if "qe" in args.comet_model_path:
            data = [
                {"src": sources[i], "mt": predictions[i],}
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

    do_pairwise_ranking = (
        args.model_name_or_path and "prometheus" in args.model_name_or_path
    )
    do_pairwise_ranking = False
    if do_pairwise_ranking:
        from vllm import LLM, SamplingParams

        sampling_params = SamplingParams(
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            best_of=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            use_beam_search=(not args.do_sample and args.num_beams > 1),
            skip_special_tokens=True,
            ignore_eos=True,
            stop=["\n\nQ:", "\n\n###", "\nProblem:"],
        )
        llm = LLM(
            model=args.model_name_or_path,
            dtype="half",
            quantization="fp8" if "8x7b" in args.model_name_or_path else None,
            max_model_len=4096,
            enforce_eager=True,
            trust_remote_code=True,
            gpu_memory_utilization=0.9,
        )
    count = 0
    for key in dico:
        if len(dico[key]) != 2:
            continue
        output_filename = f"{key}.jsonl"
        os.makedirs(os.path.join(args.data_dir, "scores"), exist_ok=True)
        output_path = os.path.join(
            os.path.join(args.data_dir, "scores"), output_filename
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
                    data2 = [
                        {"src": sources[i], "mt": pred2[i]} for i in range(len(pred2))
                    ]
                model_output = model.predict(data)
                comet_score = model_output.system_score
                model_output2 = model.predict(data2)
                comet_score2 = model_output2.system_score

                count += 1
                print(
                    f"BLASER {count}: {key}\nBLEU = {b}\nchrF++ = {c}\nCOMET = {comet_score}\n"
                )
                print(
                    f"Random {count}: {key}\nBLEU = {b2}\nchrF++ = {c2}\nCOMET = {comet_score2}\n"
                )
            except Exception as e:
                print(f"An error occurred: {e}")

        if do_pairwise_ranking:
            # Relative Grading (Pairwise Ranking)
            A = dico[key][0]
            B = dico[key][1]
            criterion = f"""
    Evaluate the quality of the translation for the sentence given as instruction based on the following criteria:
    - Length of the translation: Is the translation excessively brief (e.g. empty), overly lengthy, or appropriately concise?
    - Language of the translation: Is the translation written in the correct language (here, {target}) as intended?
    - Grammatical correctness: Does the translation adhere to established grammatical rules and conventions?
    - Meaning: Does the translation accurately convey the same meaning as the original sentence?
            """
            from fastchat.conversation import get_conv_template

            prompts = []
            for i in range(len(A)):
                prompt = get_prompt(
                    orig_instruction=sources[i],
                    orig_response_A=A[i],
                    orig_response_B=B[i],
                    orig_reference_answer=targets[i],
                    orig_criteria=criterion,
                )
                conv = get_conv_template("mistral")
                conv.set_system_message(
                    "You are a fair judge assistant tasked with providing clear, objective feedback based on specific criteria, ensuring each assessment reflects the absolute standards set for performance."
                )
                conv.append_message(conv.roles[0], prompt)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                prompts.append(prompt)
            start = 0
            if os.path.exists(output_path):
                with open(output_path, "r") as fin:
                    for line in fin:
                        start += 1
            for i in range(start, len(A), args.request_batch_size):
                inputs = prompts[i : i + args.request_batch_size]
                response = llm.generate(inputs, sampling_params)
                if args.verbose:
                    print("===")
                    for j, r in enumerate(response):
                        for element in r.outputs:
                            print(f"{j+1} -> {element.text}\n{element.finish_reason}")
                    print("===")
                outputs = [[element.text for element in r.outputs] for r in response]
                outputs = [output[0] for output in outputs]

                trigger = "[RESULT]"
                answers = []
                for j, output in enumerate(outputs):
                    if trigger in output:
                        answer = output[output.find(trigger) + len(trigger) :].strip()
                        answer = answer.split("\n")[0].split(" ")[0]
                    else:
                        answer = output
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
