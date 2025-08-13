import os
import json
import argparse
from tqdm import tqdm
from typing import List, Tuple
import numpy as np
import warnings

from comptra.data.dataset import get_datasets
from comptra.sampler import *
from comptra.prompts.templates import get_template


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        help="Identifier of the dataset",
        default="flores",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the model used for text generation.",
    )
    parser.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        help="Name or path of the tokenizer of the model used for text generation",
    )
    parser.add_argument("--src", type=str, help="Source language (e.g. English)")
    parser.add_argument("--tgt", help="Target language (e.g. French)")
    parser.add_argument(
        "--max_samples", type=int, help="Maximum number of sentences to translate."
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature of the generation."
    )
    parser.add_argument(
        "--template_key",
        type=int,
        help="Name of the template we use for ICL.",
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
    parser.add_argument("--max_new_tokens", type=int, default=75)
    parser.add_argument(
        "--max_new_tokens_divide",
        type=int,
        default=1500,
        help="Maximum number of tokens to generate when splitting a sentence.",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory.")
    parser.add_argument("--metadata_dir", type=str, help="Metadata directory.")
    parser.add_argument(
        "--inference_api",
        type=str,
        default="vllm",
        choices=["vllm", "openai", "hf", "anthropic", "cohere"],
        help="Which API to use for text generation, set to vllm by default.",
    )
    parser.add_argument("--api_key", type=str, help="OPENAI API KEY.")
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=4,
        help="Batch size for text generation.",
    )
    parser.add_argument(
        "--k",
        type=int,
        help="Number of example demonstrations for Machine Translation.",
    )
    parser.add_argument("--seed", type=int, help="Seed parameter")
    parser.add_argument(
        "--number_of_subproblems", type=int, help="Number of subproblems."
    )
    parser.add_argument("--steps", type=int, help="Number of splitting.")
    parser.add_argument("--verbose", action="store_true", help="Verbose.")

    parser.add_argument(
        "--number_of_refining_steps",
        type=int,
        default=0,
        help="Number of refining steps",
    )
    parser.add_argument(
        "--refine_after_merge",
        type=int,
        help="Number of refining steps to apply after merging.",
    )
    parser.add_argument(
        "--method_divide",
        type=str,
        choices=[
            "identity",
            "sentence",
            "space",
            "llm",
            "char",
            "structural",
            "keyword",
            "structure",
        ],
        help="How to divide the sentence into simpler propositions.",
    )
    parser.add_argument(
        "--mode_divide",
        type=str,
        help="Which few-shot prompt to use for the decomposition. e.g. vanilla, paraphrase, paraphrase-3.",
    )
    parser.add_argument(
        "--number_of_repetitions",
        type=int,
        default=4,
        help="If you divide with the `identity` method, specify how many times you want to repeat the sentence.",
    )
    parser.add_argument(
        "--merge_prompt",
        type=str,
        choices=["refine", "norefine", "vanilla"],
        help="How to merge the translation of simpler propositions into the whole sentence translation.",
    )
    parser.add_argument(
        "--method_translate",
        type=str,
        # choices=["nllb", "vanilla", "cot", "maps"],
        help="How to translate the simpler propositions.",
    )
    parser.add_argument(
        "--selection_method",
        type=str,
        choices=["greedy", "bleu", "blaser", "comet-qe"],
        help="How to select the best translation out of multiple generations.",
    )
    parser.add_argument(
        "--nllb_name_or_path",
        type=str,
        default="facebook/nllb-200-distilled-600M",
        help="Name or path to the NLLB model.",
    )
    parser.add_argument(
        "--retriever_type",
        type=str,
        help="Type of retriever for in-context example selection, e.g. 'Random', 'bm25s' etc.",
    )
    parser.add_argument(
        "--number_of_merge_demonstrations",
        type=int,
        help="Number of demonstrations to use during the merge operation",
    )
    parser.add_argument(
        "--base_model_name_or_path",
        type=str,
        help="Name or path of the base model associated to the LoRA module used for text generation.",
    )
    parser.add_argument(
        "--lora_rank", type=int, default=16, help="The lora rank parameter."
    )
    parser.add_argument(
        "--enable_lora",
        action="store_true",
        help="Whether the model_name_or_path is a LoRA fine-tuning. In this case you have to specify --base_model_name_or_path.",
    )
    return parser.parse_args()


def main(args):
    print(
        f"Model name : {args.model_name_or_path}\nTranslation from {args.src} to {args.tgt}."
    )
    try:
        template = get_template(key=args.template_key, src=args.src, tgt=args.tgt)
    except:
        warnings.warn(
            f"The template key provided ({args.template_key}) is not supported! We'll be using a default template."
        )
        template = None

    if (
        "gemma-3" in args.model_name_or_path
        and "checkpoint-" in args.model_name_or_path
    ) or ("nllb" in args.model_name_or_path):
        from transformers import AutoTokenizer, AutoProcessor

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        tokenizer.save_pretrained(args.model_name_or_path)
        try:
            processor = AutoProcessor.from_pretrained(args.tokenizer_name_or_path)
            processor.save_pretrained(args.model_name_or_path)
        except Exception as e:
            print(f"Exception: {e}")
            
    if args.dataset_name_or_path not in ["topxgen"]:
        dataset_src = get_datasets(args.dataset_name_or_path, args.src)
        dataset_tgt = get_datasets(args.dataset_name_or_path, args.tgt)
    else:
        dataset_src, dataset_tgt = get_datasets(args.dataset_name_or_path, args.tgt if args.tgt != "English" else args.src)
        if args.src != "English":
            dataset_src, dataset_tgt = dataset_tgt, dataset_src
    # Check if the datasets are empty
    if len(dataset_src["dev"]) == 0 or len(dataset_tgt["dev"]) == 0:
        raise ValueError(
            f"The datasets are empty! Please check the dataset name or path: {args.dataset_name_or_path}"
        )
    # Check if the datasets have the same number of examples
    if len(dataset_src["dev"]) != len(dataset_tgt["dev"]):
        raise ValueError(
            f"The datasets have different number of examples! Please check the dataset name or path: {args.dataset_name_or_path}"
        )
    # Check if the datasets have the same number of examples
    if len(dataset_src["devtest"]) != len(dataset_tgt["devtest"]):
        raise ValueError(
            f"The datasets have different number of examples! Please check the dataset name or path: {args.dataset_name_or_path}"
        )

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": args.src,
        "tgt": args.tgt,
        "template": template,
        "merge_prompt": args.merge_prompt,
        "selection_method": args.selection_method,
        "method_translate": args.method_translate,
        "nllb_name_or_path": args.nllb_name_or_path,
        "method_divide": args.mode_divide if args.mode_divide else args.src,
    }

    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": args.num_return_sequences,
        "num_beams": args.num_beams,
        "do_sample": args.do_sample,
        "request_batch_size": args.request_batch_size,
        "verbose": args.verbose,
    }
    
    print(f"generation_kwargs: {generation_kwargs}")

    if args.enable_lora and args.base_model_name_or_path:
        print(f"Evaluating a LoRa module.")
        arguments["model_name_or_path"] = args.base_model_name_or_path
        arguments["enable_lora"] = True
        arguments["max_lora_rank"] = args.lora_rank
        arguments["lora_path"] = args.model_name_or_path

    if args.inference_api == "vllm":
        sampler = vLLMSampler(**arguments)
    elif args.inference_api == "openai":
        sampler = OpenAISampler(api_key=args.api_key, **arguments)
    elif args.inference_api == "anthropic":
        sampler = AnthropicSampler(**arguments)
    elif args.inference_api == "cohere":
        sampler = cohereSampler(**arguments)
    elif args.inference_api == "hf":
        sampler = HFSampler(**arguments)
    else:
        sampler = Sampler(**arguments)

    # Get the decomposition function
    if args.method_divide == "identity":

        def divide_fn(sentences: List[str], n_splits: int) -> List[List[str]]:
            return [[sentence] * args.number_of_repetitions for sentence in sentences]

    elif args.method_divide == "llm":

        def divide_fn(sentences: List[str], n_splits: int) -> List[List[str]]:
            generation_kwargs["max_new_tokens"] = args.max_new_tokens_divide
            return sampler.divide(
                sentences=sentences, n_splits=n_splits, **generation_kwargs
            )

    elif args.method_divide == "structural":
        from comptra.prompts.decompose import structural

        def divide_fn(sentences: List[str], n_splits: int) -> List[List[str]]:
            return [structural(sentence, n_splits=n_splits) for sentence in sentences]

    elif args.method_divide == "sentence":
        from comptra.prompts.decompose import sentence_split

        def divide_fn(sentences: List[str], n_splits: int) -> List[List[str]]:
            return [
                sentence_split(sentence, n_splits=n_splits) for sentence in sentences
            ]

    elif args.method_divide == "space":
        from comptra.prompts.decompose import equal_split

        def divide_fn(sentences: List[str], n_splits: int) -> List[List[str]]:
            return [equal_split(sentence, n_splits=n_splits) for sentence in sentences]

    elif args.method_divide == "char":
        from comptra.prompts.decompose import characterwise_split

        def divide_fn(sentences: List[str], n_splits: int) -> List[List[str]]:
            return [
                characterwise_split(sentence, n_splits=n_splits)
                for sentence in sentences
            ]

    elif args.method_divide == "keyword":
        from comptra.prompts.decompose import keyword_splitting

        def divide_fn(sentences: List[str], n_splits: int) -> List[List[str]]:
            return [
                keyword_splitting(sentence, n_splits=n_splits) for sentence in sentences
            ]

    else:
        raise KeyError(f"Unsupported divide method {args.method_divide}.")

    # Get the translation function
    def translate_fn(
        sentences: List[str], demonstrations: List[List[Tuple[str, str]]] = None
    ) -> List[str]:
        generation_kwargs["max_new_tokens"] = args.max_new_tokens
        return sampler.translate(
            sentences=sentences, demonstrations=demonstrations, **generation_kwargs
        )

    # Get the merge function
    def merge_fn(
        sentences: List[str],
        inputs: List[List[str]],
        outputs: List[List[str]],
        demonstrations: List[List[Tuple[str, str]]] = None,
    ) -> List[str]:
        generation_kwargs["max_new_tokens"] = max(1500, args.max_new_tokens)
        return sampler.merge(
            sentences=sentences,
            inputs=inputs,
            outputs=outputs,
            demonstrations=demonstrations,
            **generation_kwargs,
        )

    # Get the refine function
    def refine_fn(
        sources: List[str],
        prev_translations: List[str],
        max_tokens: int,
        number_of_refining_steps: int,
    ):
        generation_kwargs["max_new_tokens"] = max(500, max_tokens)
        return sampler.refine(
            sentences=sources,
            prev_translations=prev_translations,
            number_of_refining_steps=number_of_refining_steps,
            **generation_kwargs,
        )

    # Create output files and directories
    if args.metadata_dir:
        metadata_dir = args.metadata_dir
    else:
        depth = args.steps
        breadth = args.number_of_subproblems
        if breadth >= 0:
            metadata_dir = f"{args.src}_to_{args.tgt}_{args.k}_shot_seed_{args.seed}_{args.method_divide}_{args.merge_prompt}_{breadth}_{depth}_{args.number_of_refining_steps}"
        else:
            metadata_dir = f"{args.src}_to_{args.tgt}_{args.k}_shot_seed_{args.seed}_{args.method_divide}_{args.merge_prompt}_None_{depth}_{args.number_of_refining_steps}"

    output_dir = args.output_dir
    output_dir = os.path.join(output_dir, args.model_name_or_path.split("/")[-1])
    os.makedirs(output_dir, exist_ok=True)

    metadata_dir = os.path.join(output_dir, metadata_dir)
    os.makedirs(metadata_dir, exist_ok=True)

    if args.k is not None and args.k > 0:
        print(f"Translation with {args.k} demonstrations!")
        rng = np.random.default_rng(args.seed)
        if args.retriever_type == "Random":
            print(f"We use random selection!")
            indices = [
                rng.choice(len(dataset_src["dev"]), size=args.k, replace=False).tolist()
                for _ in range(len(dataset_src["devtest"]))
            ]
            list_of_demonstrations = []
            for i in range(len(dataset_src["devtest"])):
                a = dataset_src["dev"].select(indices[i])
                b = dataset_tgt["dev"].select(indices[i])
                demonstrations = [
                    (a[j]["sentence"], b[j]["sentence"]) for j in range(args.k)
                ]
                list_of_demonstrations.append(demonstrations)
        else:
            print(f"We select with {args.retriever_type}.")
            from comptra.retriever import Retriever

            retriever = Retriever(
                dataset_name_or_path=args.dataset_name_or_path,
                retriever_type=args.retriever_type,
                source_language=args.src,
                target_language=args.tgt,
                ds_src=dataset_src,
                ds_tgt=dataset_tgt,
                path=output_dir,
            )

    else:
        list_of_demonstrations = None

    if (
        args.number_of_merge_demonstrations is not None
        and args.number_of_merge_demonstrations > 0
    ):
        print(f"Merge with {args.number_of_merge_demonstrations} demonstrations!")

    # Get the sentences to translate
    source_sentences = [
        example["sentence"]
        for i, example in enumerate(dataset_src["devtest"])
        if args.max_samples is not None and i < args.max_samples
    ]
    print(f"There are {len(source_sentences)} samples!")
    # source sentences, sentences for round 1, ..., sentences for round N
    list_of_sentences = [source_sentences]
    # parents, parents for round 1, ..., parents for round N
    list_of_dictionaries = [{}]
    # Go through the number of dividing rounds
    for round in tqdm(range(args.steps)):
        list_of_propositions = []
        if os.path.exists(os.path.join(output_dir, f"divide_{round+1}.jsonl")):
            assert os.path.exists(
                os.path.join(output_dir, f"parent_{round+1}.jsonl")
            ), f"divide_{round+1}.jsonl exists but not parent_{round+1}.jsonl"
            print(
                f"Reading from {os.path.join(output_dir, f'divide_{round+1}.jsonl')}!"
            )
            with open(os.path.join(output_dir, f"divide_{round+1}.jsonl"), "r") as fin:
                for line in fin:
                    list_of_propositions.append(json.loads(line)["propositions"])
            dico = {}
            with open(os.path.join(output_dir, f"parent_{round+1}.jsonl"), "r") as fin:
                for line in fin:
                    dico = json.loads(line)
                    break
            # Number of keys in dico indicates the number of sentences that have already been divided
            print(dico)
            start = 1 + max([v for _, v in dico.items()]) if len(dico) != 0 else 0
            print(f"Resuming from index {start}.")
        else:
            start = 0
            print(f"Start from the beginning ({start}).")
        # Resume the division were it stopped
        sentences = list_of_sentences[-1]
        resume_list_of_propositions = divide_fn(
            sentences[start:], n_splits=args.number_of_subproblems
        )
        list_of_propositions.extend(resume_list_of_propositions)
        with open(os.path.join(output_dir, f"divide_{round+1}.jsonl"), "a") as fout:
            for j in range(start, len(sentences)):
                fout.write(
                    json.dumps(
                        {
                            "sentence": sentences[j],
                            "propositions": list_of_propositions[j],
                        }
                    )
                    + "\n"
                )
        dico = {}
        key = 0
        for a, propositions in enumerate(list_of_propositions):
            for b in range(len(propositions)):
                dico[key + b] = a
            key += len(propositions)
        with open(os.path.join(output_dir, f"parent_{round+1}.jsonl"), "w") as fout:
            fout.write(json.dumps(dico))

        # Sentences for the next round are the propositions of the current round
        sentences = [
            prop for propositions in list_of_propositions for prop in propositions
        ]
        list_of_sentences.append(sentences)
        list_of_dictionaries.append(dico)

    step = args.steps
    previous_translations = None
    while step >= 0:
        sentences = list_of_sentences[step]
        # Resume where we stopped
        current_translations = []
        if os.path.exists(os.path.join(metadata_dir, f"translate_{step}.jsonl")):
            with open(
                os.path.join(metadata_dir, f"translate_{step}.jsonl"), "r"
            ) as fin:
                for line in fin:
                    current_translations.append(json.loads(line)["translation"])

        start = len(current_translations)
        if step == args.steps:
            # We are at the leaves of the tree, we should translate each sentence
            for i in tqdm(range(start, len(sentences), args.request_batch_size)):
                inputs = sentences[i : min(i + args.request_batch_size, len(sentences))]
                batch_of_demonstrations = []
                for j in range(len(inputs)):
                    # In-context examples
                    if args.k is None or args.k == 0:
                        # Zero-shot MT
                        demonstrations = []
                    else:
                        if args.retriever_type == "Random":
                            # Random case, all the subsentences of the same sentence use the same
                            # Should find the super parent of each sentence
                            super_parent = (
                                i + j
                            )  # i + j = index of the sentence in List[str] (sentences)
                            for m in range(step, -1, -1):
                                super_parent = list_of_dictionaries[m].get(
                                    super_parent, super_parent
                                )
                            assert super_parent < len(
                                list_of_sentences[0]
                            ), f"The super parent {super_parent} is greater than or equal to {len(list_of_sentences[0])}."
                            demonstrations = list_of_demonstrations[super_parent]
                        else:
                            demonstrations = retriever.query(
                                sentence=inputs[j],
                                k=args.k,
                                idx_sentence=(
                                    i + j if (args.retriever_type == "SONAR") else None
                                ),
                                level=step,
                            )

                    batch_of_demonstrations.append(demonstrations)
                # Translate the sentences
                outputs = translate_fn(
                    sentences=inputs, demonstrations=batch_of_demonstrations
                )
                # Refining
                if args.number_of_refining_steps:
                    print("Refining after translate.")
                    outputs = refine_fn(
                        inputs,
                        outputs,
                        max_tokens=max(1500, args.max_new_tokens),
                        number_of_refining_steps=args.number_of_refining_steps,
                    )
                # Save the predictions to an output file
                with open(
                    os.path.join(metadata_dir, f"translate_{step}.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as fout:
                    for j, output in enumerate(outputs):
                        current_translations.append(output.strip())
                        fout.write(
                            json.dumps(
                                {
                                    "sentence": sentences[i + j],
                                    "translation": output.strip(),
                                }
                            )
                            + "\n"
                        )
        else:
            # We are not at the leaves of the tree, we should translate each sentence based on the next level's translations i.e. merge
            assert (
                previous_translations is not None
            ), f"previous_translations ({previous_translations}) is None"
            inputs = [[] for _ in range(len(sentences))]
            outputs = [[] for _ in range(len(sentences))]
            dico = list_of_dictionaries[step + 1]
            for key in dico:
                inputs[dico[key]].append(list_of_sentences[step + 1][key])
                outputs[dico[key]].append(previous_translations[key])

            for i in tqdm(range(start, len(sentences), args.request_batch_size)):
                sentences_batch = sentences[
                    i : min(i + args.request_batch_size, len(sentences))
                ]
                inputs_batch = inputs[i : min(i + args.request_batch_size, len(inputs))]
                outputs_batch = outputs[
                    i : min(i + args.request_batch_size, len(outputs))
                ]

                outputs_list = merge_fn(
                    sentences=sentences_batch,
                    inputs=inputs_batch,
                    outputs=outputs_batch,
                    demonstrations=(
                        [
                            retriever.query(
                                sentence=sentences_batch[j],
                                k=args.number_of_merge_demonstrations,
                                idx_sentence=(
                                    i + j if (args.retriever_type == "SONAR") else None
                                ),
                                level=step,
                            )
                            for j in range(len(sentences_batch))
                        ]
                        if args.number_of_merge_demonstrations
                        else None
                    ),
                )
                # Refine the translation for a given number of steps after the merge
                if args.refine_after_merge is not None and args.refine_after_merge > 0:
                    print("Refining after merge.")
                    outputs_list = refine_fn(
                        sentences_batch,
                        outputs_list,
                        max_tokens=max(1500, args.max_new_tokens),
                        number_of_refining_steps=args.refine_after_merge,
                    )
                # Save the predictions to an output file
                with open(
                    os.path.join(metadata_dir, f"translate_{step}.jsonl"),
                    "a",
                    encoding="utf-8",
                ) as fout:
                    for j, output in enumerate(outputs_list):
                        current_translations.append(output)
                        fout.write(
                            json.dumps(
                                {
                                    "sentence": sentences_batch[j],
                                    "translation": output.strip(),
                                }
                            )
                            + "\n"
                        )
        # Update parameters
        step -= 1
        previous_translations = current_translations

    print("END")


if __name__ == "__main__":
    args = parse_args()
    main(args)
