import argparse
from datasets import load_dataset
from comptra.languages import MAPPING_LANG_TO_KEY
from comptra.prompts.templates import get_template
from comptra.sampler import *

from tqdm import tqdm
import numpy as np
import time
import json
import os
import re

from comptra.utils import _stop_at_stop_token, get_bigrams, count_bigrams, is_lang

from rouge_score import rouge_scorer
from multiprocessing import Pool
from functools import partial
from itertools import combinations
from collections import Counter

from sacrebleu.metrics import BLEU

bleu = BLEU(tokenize="flores200")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

WORD_REPETITION_THRESHOLD = 25  # 10
CHARACTER_REPETITION_THRESHOLD = 25  # 10
PARAGRAPH_COUNT = 3


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_generation_pipeline",
        type=str,
        help="Data generation pipeline e.g. topxgen, self-instruct, knn-instruct",
        default="topxgen"
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
    parser.add_argument("--api_key", type=str, help="OPENAI API KEY.")
    parser.add_argument(
        "--inference_api",
        type=str,
        default="vllm",
        help="Which API to use for text generation, set to vllm by default.",
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=4,
        help="Batch size for the generations.",
    )
    parser.add_argument(
        "--seed", type=int, default=122, help="Seed for random number generation."
    )

    parser.add_argument(
        "--max_new_tokens",
        type=int,
        help="Maximum number of tokens to generate per query.",
    )
    parser.add_argument(
        "--temperature", type=float, help="Temperature of the generation."
    )
    parser.add_argument("--top_p", type=float, help="Nucleus sampling parameter.")
    parser.add_argument("--repetition_penalty", type=float, help="Repetition penalty.")
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        help="Number of responses to return per query.",
    )
    parser.add_argument(
        "--num_beams", type=int, help="Number of beams for beams search."
    )
    parser.add_argument(
        "--do_sample",
        action="store_true",
        help="Whether to use sampling in the generation.",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Whether to write on the console."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        help="Languages to generate. A space-separated list of capitalized language names.",
    )
    parser.add_argument(
        "--seed_topics_path",
        type=str,
        help="Path to a file containing all the topics of interest.",
    )
    parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument(
        "--translate_dir",
        type=str,
        help="Name of the directory containing the translation files.",
    )
    parser.add_argument(
        "--output_filenames",
        nargs="+",
        type=str,
        help="Output filenames for each language we want to generate.",
    )
    parser.add_argument(
        "--number_of_instructions",
        type=int,
        help="Total number of instructions to generate.",
    )
    parser.add_argument(
        "--number_of_instructions_per_language",
        nargs="+",
        type=int,
        help="Minimum number of instructions to generate per language e.g. `10 20 30`",
    )
    parser.add_argument(
        "--number_of_icl_demonstrations",
        type=int,
        default=4,
        help="Number of in-context demonstrations for each text generation step.",
    )
    parser.add_argument(
        "--number_of_icl_phrases",
        type=int,
        default=0,
        help="Number of phrases in the language of interest to include in the prompt",
    )
    parser.add_argument(
        "--number_of_generations_per_step",
        type=int,
        default=2,
        help="Number of new text to generate per generation step.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Similarity threshold for the addition of new texts.",
    )
    parser.add_argument(
        "--use_nllb",
        action="store_true",
        help="Whether to use NLLB for the back-translation step.",
    )
    parser.add_argument(
        "--number_of_icl_synthetic_demonstrations",
        type=str,
        default=2,
        help="Number of synthetic demonstrations for self-instruct.",
    )
    parser.add_argument("--number_of_rounds", type=int, help="Number of rounds.")
    parser.add_argument(
        "--max_samples", type=int, help="Max seed instruction samples (for debugging)."
    )
    parser.add_argument(
        "--template_key",
        type=int,
        help="Name of the template we use for ICL.",
    )
    parser.add_argument(
        "--number_of_demonstrations",
        type=int,
        default=5,
        help="Number of in-context demonstrations for translation.",
    )
    parser.add_argument(
        "--nllb_name_or_path",
        type=str,
        help='Name or path to the nllb model used for BT.'
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default="English",
        help="The target language for back-translation."
    )
    return parser.parse_args()


def get_prompt(
    language: str, number: int, topic: str, examples: str = None, phrases: str = None
):
    GENERATE_PROMPT = f"""
You are an helpful and precise polyglot assistant which is highly proficient in multiple languages including {language}.
""".strip()
    if examples:
        GENERATE_PROMPT += (
            "\n"
            + f"""
Here are a few examples of paragraphs written in different languages that you can create.

<Examples>
{examples}
</Examples>
""".strip()
            + "\n"
        )
    else:
        GENERATE_PROMPT += "\n\n"
    GENERATE_PROMPT += f"""
Write {number} diverse, informative and insightful texts (at most {PARAGRAPH_COUNT} paragraphs) in {language} within the context of the following topic(s): {topic}.
Your texts should delve into the nuance of the topic(s), offering fresh perspectives and deeper analysis.

Aim to:
- Inform: Provide valuable, well-researched information that educates the reader.
- Engage: Write in a conversational tone that connects with the audience, making complex ideas accessible.
- Illustrate: Use examples, anecdotes, or personal experiences to bring the topic(s) to life. Do not give a title and do not start with sentences like "Have you ever..." or "Hello dear readers...", simply write the content without these introductory phrases.

Moreover,
- Include relevant words/concepts: Ensure the correct use of vocabulary in {language} relevant to the topic's semantic field.
- Write with care: Ensure the grammatical accuracy when you write in {language}.
- DO NOT translate the texts provided as examples.
- DO NOT give titles to the texts you write.
""".rstrip()
    if phrases:
        GENERATE_PROMPT += f"\n\nFor example, here are high-quality sentences written in {language}\n<Sentences>\n{phrases}\n</Sentences>\n"
    else:
        GENERATE_PROMPT += "\n"
    GENERATE_PROMPT += (
        "\n"
        + f"""
Your answer should have the following structure:
1. <text 1>
...
{number}. <text {number}>
i.e. a numbered list.

Go ahead and write the texts.
""".strip()
    )
    return GENERATE_PROMPT


def main(args):
    """
    Arguments
    ---------
    """
    rng = np.random.default_rng(args.seed)
    languages = args.languages

    dico_of_example_sentences = {}
    for language in languages:
        dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])["dev"]
        dico_of_example_sentences[language] = (
            [example["sentence"] for example in dataset]
            if (args.max_samples is None or args.max_samples < 0)
            else [example["sentence"] for example in dataset][: args.max_samples]
        )

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": languages[0],
        "tgt": "English",
        "template": get_template(key=template_key, src=languages[0], tgt="English"),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
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

    MAPPING_LANG_TO_ID = {
        "Arabic": "ar",
        "English": "en",
        "Chinese": "zh",
        "German": "de",
        "Greek": "el",
        "Hindi": "hi",
        "Romanian": "ro",
        "Russian": "ru",
        "Spanish": "es",
        "Thai": "th",
        "Turkish": "tr",
        "Vietnamese": "vi",
    }

    dico_of_seed_instructions = {}
    for k, v in MAPPING_LANG_TO_ID.items():
        dataset = load_dataset("google/xquad", f"xquad.{v}")
        indices = []
        L = len(dataset["validation"])
        i = 0
        while i < L:
            indices.append(i)
            example = dataset["validation"][i]
            j = i + 1
            next_example = dataset["validation"][j]
            while (
                j < L and example["context"].strip() == next_example["context"].strip()
            ):
                j += 1
                if j < L:
                    next_example = dataset["validation"][j]
            i = j
        print(f"{k}: {len(indices)}.")
        dico_of_seed_instructions[k] = [
            example["context"] for example in dataset["validation"].select(indices)
        ]

    if args.seed_topics_path.endswith(".jsonl"):
        seed_topics = [json.loads(l) for l in open(args.seed_topics_path, "r")]
    else:
        seed_topics = open(args.seed_topics_path, "r").read().split("\n")
        seed_topics = [topic.strip() for topic in seed_topics if topic.strip()]

    print(
        f"There are {len(seed_topics)} topics. The first one is: {seed_topics[0]}. The last one is: {seed_topics[-1]}."
    )

    if "wiki" not in args.seed_topics_path.lower():
        seed_topics = [
            (
                topic
                if len(topic.split(", ")) == 1
                else ", ".join(topic.split(", ")[:-1]) + " and " + topic.split(", ")[-1]
            )
            for topic in seed_topics
        ]

    os.makedirs(args.output_dir, exist_ok=True)

    output_filenames = (
        args.output_filenames
        if args.output_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(output_filenames) == len(
        languages
    ), f"The number of output filenames ({len(output_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_machine_instructions = {language: [] for language in languages}
    dico_of_topics = {language: [] for language in languages}
    number_of_instructions = 0

    for i, output_filename in enumerate(output_filenames):
        if os.path.exists(os.path.join(args.output_dir, output_filename)):
            with open(os.path.join(args.output_dir, output_filename), "r") as fin:
                for line in fin:
                    dico_of_machine_instructions[languages[i]].append(
                        json.loads(line)["text"]
                    )
                    number_of_instructions += 1
                    try:
                        dico_of_topics[languages[i]].append(json.loads(line)["topic"])
                    except Exception as e:
                        pass

    # Tokenize
    tokenized_machine_instructions = {
        key: [scorer._tokenizer.tokenize(inst) for inst in value]
        for key, value in dico_of_machine_instructions.items()
    }
    # Progress bar
    progress_bar = tqdm(total=args.number_of_instructions)
    progress_bar.update(number_of_instructions)
    # Number of cycles without adding data
    repetition = 0
    while number_of_instructions < args.number_of_instructions or (
        args.number_of_instructions_per_language is not None
        and any(
            len(dico_of_machine_instructions[languages[i]])
            < args.number_of_instructions_per_language[i]
            for i in range(len(languages))
        )
    ):
        if repetition >= 50:
            print(f"Too many rounds without adding data ({repetition}). Stopping.")
            break
        batched_topics = rng.choice(
            seed_topics, size=args.request_batch_size, replace=False
        ).tolist()
        idx_language = rng.choice(len(languages), size=1).tolist()[0]
        language = languages[idx_language]
        prompts = []
        for i in range(args.request_batch_size):
            # Choose the in-context demonstrations
            icl_languages = rng.choice(
                list(MAPPING_LANG_TO_ID.keys()),
                size=args.number_of_icl_demonstrations,
                replace=False,
            ).tolist()
            indices = rng.choice(
                len(dico_of_seed_instructions["English"]),
                size=args.number_of_icl_demonstrations,
                replace=False,
            ).tolist()
            demonstrations = [
                dico_of_seed_instructions[lan][idx]
                for (lan, idx) in zip(icl_languages, indices)
            ]
            examples = "\n\n".join(
                f"{j+1}. {icl_lan} text\n{demonstration}"
                for j, (icl_lan, demonstration) in enumerate(
                    zip(icl_languages, demonstrations)
                )
            )
            # choose the example phrases in language
            if (
                args.number_of_icl_phrases is not None
                and args.number_of_icl_phrases > 0
            ):
                phrases = rng.choice(
                    dico_of_example_sentences[language],
                    size=args.number_of_icl_phrases,
                    replace=False,
                ).tolist()
            else:
                phrases = []

            phrases = "\n".join(
                [f"{j+1}. {phrase}" for (j, phrase) in enumerate(phrases)]
            )

            prompts.append(
                get_prompt(
                    language=language,
                    number=args.number_of_generations_per_step,
                    topic=batched_topics[i],
                    examples=examples,
                    phrases=phrases,
                )
            )
            if args.verbose:
                print(
                    f"Generating {language} based on {', '.join(icl_languages[:-1]) + ' and ' + icl_languages[-1]} about {batched_topics[i]}"
                )
                if i == 0:
                    print(
                        f"{'-'*75}\nPrompt number {i+1}.\n{'-'*75}\n{sampler.apply_chat_template(prompts[-1])}\n{'-'*75}"
                    )
        try:
            answers = sampler.generate(
                [sampler.apply_chat_template(prompt) for prompt in prompts],
                **generation_kwargs,
            )
        except Exception as e:
            print(f"Exception during generation: {e}")
            answers = []
        new_instructions = []
        new_topics = []
        number_of_candidate_instructions = 0
        for r, answer in enumerate(answers):
            answer = answer[0]
            answer = _stop_at_stop_token(
                answer, [f"{args.number_of_generations_per_step + 1}.\t"] + STOP_WORDS
            )
            if "1. \n" in answer:
                pattern = r"(\d+\. \n)"
            elif "1. " in answer:
                # pattern = r"(\d+\. )"
                pattern = r"(\d+\. ?(?:\n)?)"
            else:
                pattern = r"(\d+\.\n)"
            splitted_answer = re.split(pattern, answer)
            try:
                assert (
                    len(splitted_answer) == 2 * args.number_of_generations_per_step + 1
                ), "There is an issue."
            except Exception as exc:
                print(
                    f"The generation does not have the required quality. Pattern = {pattern}, length = {len(splitted_answer)}"
                )
                continue
            candidates = []
            for j, element in enumerate(splitted_answer):
                if j == 0:
                    # everything before the first match
                    continue
                if j % 2 == 1:
                    # matches the iterator
                    continue
                if len(element.strip().split("\n")) >= 2:
                    sentences = element.strip().split("\n")
                    """
                    idx = len(sentences) - 1
                    while idx >= 0:
                        candidate = sentences[idx]
                        if candidate.strip() != "":
                            break
                        idx -= 1
                    """
                    candidate = " ".join(sentences)
                else:
                    candidate = element.strip()
                candidate = " ".join(
                    [
                        phrase.strip()
                        for phrase in candidate.split("\n")
                        if phrase.strip() != ""
                    ]
                )
                print(f"{number_of_candidate_instructions + j//2}. {candidate}")
                # Check for word-level and character-level repetitions
                if len(candidate.split(" ")) <= 2:
                    continue
                bigrams = count_bigrams(get_bigrams(candidate))
                bigram, highest_frequency = bigrams.most_common()[0]
                if highest_frequency >= PARAGRAPH_COUNT * WORD_REPETITION_THRESHOLD:
                    print(
                        f"More than {PARAGRAPH_COUNT * WORD_REPETITION_THRESHOLD} repeating bigrams."
                    )
                    continue
                longest_word = max(candidate.split(" "), key=lambda x: len(x))
                longest_word = " ".join(list(longest_word))
                bigrams = count_bigrams(get_bigrams(longest_word))
                bigram, highest_frequency = bigrams.most_common()[0]
                if (
                    highest_frequency
                    >= PARAGRAPH_COUNT * CHARACTER_REPETITION_THRESHOLD
                ):
                    print(
                        f"More than {PARAGRAPH_COUNT * CHARACTER_REPETITION_THRESHOLD} repeating bicharacters."
                    )
                    continue
                skip = False
                for a, b in Counter(
                    list(combinations(candidate.split(" "), 2))
                ).most_common():
                    if b < 2 * PARAGRAPH_COUNT * WORD_REPETITION_THRESHOLD:
                        break
                    if len(a[0]) >= 5 and len(a[1]) >= 5:
                        print(f"{a} occurs {b} times.")
                        skip = True
                        break
                if skip:
                    continue
                candidate = candidate.replace("<text 1>", "").replace("<text 2>", "")
                candidates.append(candidate)
            number_of_candidate_instructions += len(candidates)
            new_instructions.append(candidates)
            new_topics.append([batched_topics[r]] * len(candidates))

        kept_instructions = []
        kept_topics = []
        number_of_kept_instructions = 0
        start_cleaning = time.time()
        for r, candidates in enumerate(new_instructions):
            for topic, candidate in zip(new_topics[r], candidates):
                candidate_tokens = scorer._tokenizer.tokenize(candidate)
                with Pool(4) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, candidate_tokens),
                        tokenized_machine_instructions[language],
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                if len(rouge_scores) > 0 and max(rouge_scores) > args.threshold:
                    continue

                dico_of_machine_instructions[language].append(candidate)
                dico_of_topics[language].append(topic)
                tokenized_machine_instructions[language].append(candidate_tokens)
                # Kept instructions
                number_of_kept_instructions += 1
                kept_instructions.append(candidate)
                kept_topics.append(topic)
                progress_bar.update(1)
        print(
            f"We keep {number_of_kept_instructions}/{number_of_candidate_instructions} intructions, in {(time.time() - start_cleaning):.2f}s"
        )
        assert len(kept_instructions) == len(
            kept_topics
        ), f"The number of kept instructions ({len(kept_instructions)}) should match the number of kept topics ({len(kept_topics)})."
        with open(
            os.path.join(args.output_dir, output_filenames[idx_language]),
            "a",
            encoding="utf-8",
        ) as fout:
            for r, instruction in enumerate(kept_instructions):
                fout.write(
                    json.dumps(
                        {"text": instruction, "topic": kept_topics[r]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

        number_of_instructions += len(kept_instructions)
        if number_of_kept_instructions == 0:
            repetition += 1
        else:
            repetition = 0

    print("END GENERATION")
    print(
        {
            language: len(dico_of_machine_instructions[language])
            for language in dico_of_machine_instructions
        }
    )
    print("START TRANSLATION")
    if args.use_nllb:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from accelerate import Accelerator
        import torch

        if not args.nllb_name_or_path:
            nllb_name_or_path = "facebook/nllb-200-3.3B"
        else:
            nllb_name_or_path = args.nllb_name_or_path

        translator = AutoModelForSeq2SeqLM.from_pretrained(
            nllb_name_or_path,
            device_map=(
                {"": Accelerator().process_index} if torch.cuda.is_available() else None
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(nllb_name_or_path)
    # Sentence splitter
    from sentence_splitter import SentenceSplitter

    splitter = SentenceSplitter(language="en")
    # Blaser
    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    from sonar.models.blaser.loader import load_blaser_model

    blaser = load_blaser_model("blaser_2_0_qe").eval()
    text_embedder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
    )
    # Let's start
    if args.translate_dir:
        path_to_translations = os.path.join(args.output_dir, args.translate_dir)
    else:
        if args.use_nllb:
            path_to_translations = os.path.join(args.output_dir, nllb_name_or_path.split("/")[-1])
        else:
            path_to_translations = os.path.join(
                args.output_dir, args.model_name_or_path.split("/")[-1]
            )

    os.makedirs(path_to_translations, exist_ok=True)
    filenames = [f"{language}_translate.jsonl" for language in languages]

    for i, filename in tqdm(enumerate(filenames)):
        print(f"Translating from {languages[i]}.")
        if args.use_nllb:
            tokenizer.src_lang = MAPPING_LANG_TO_KEY[languages[i]]
        else:
            from comptra.retriever import Retriever

            retriever = Retriever(
                source_language=languages[i],
                dataset_name_or_path="flores",
                retriever_type="bm25s",
                target_language=args.target_language,
            )
            sampler.update_template(
                get_template(key=template_key, src=languages[i], tgt=args.target_language)
            )
            sampler.update_src(languages[i])
            print(f"sampler.src: {sampler.src}")
            print(f"sampler.template: {sampler.template}")
            print(f"beam size: {generation_kwargs['num_beams']}")
            generation_kwargs["do_sample"] = False
            generation_kwargs["temperature"] = 0.0
            generation_kwargs["max_new_tokens"] = 500

        start = 0
        if os.path.exists(os.path.join(path_to_translations, filename)):
            with open(os.path.join(path_to_translations, filename), "r") as fin:
                for _ in fin:
                    start += 1
        print(f"Resuming at index {start}.")
        instructions = dico_of_machine_instructions[languages[i]]
        topics = dico_of_topics[languages[i]]
        for j in range(start, len(instructions), args.request_batch_size):
            batch_of_instructions = instructions[j : j + args.request_batch_size]
            batch_of_topics = topics[j : j + args.request_batch_size]
            list_of_batch_of_sentences = []
            for instruction in batch_of_instructions:
                paragraph = _stop_at_stop_token(instruction, STOP_WORDS)
                if languages[i] == "Nepali":
                    c = "।"
                    sentences = paragraph.split(f"{c} ")
                    sentences = [element.strip() for element in sentences]
                    sentences = [
                        element + c if not element.endswith(c) else element
                        for element in sentences
                    ]
                elif languages[i] == "Urdu":
                    c = "۔"
                    sentences = paragraph.split(f"{c} ")
                    sentences = [element.strip() for element in sentences]
                    sentences = [
                        element + c if not element.endswith(c) else element
                        for element in sentences
                    ]
                else:
                    sentences = splitter.split(text=paragraph)
                # Put the sentences together
                batch_of_sentences = [
                    sentence
                    for sentence in sentences
                    if is_lang(sentence, languages[i])
                ]
                # Filter too short sentences
                batch_of_sentences = [
                    sentence for sentence in batch_of_sentences if len(sentence) >= 40
                ]
                list_of_batch_of_sentences.append(batch_of_sentences)
            # Flatten the list of sentences
            flat_list_of_batch_of_sentences = [
                sentence
                for batch_of_sentences in list_of_batch_of_sentences
                for sentence in batch_of_sentences
            ]
            if len(flat_list_of_batch_of_sentences) != 0:
                continue
            # Translate
            if args.use_nllb:
                inputs = tokenizer(
                    flat_list_of_batch_of_sentences, return_tensors="pt", padding=True
                ).to(translator.device)
                # Output length
                output_len = 2 * inputs["attention_mask"].sum(-1).max().item()
                translated_tokens = translator.generate(
                    **inputs,
                    forced_bos_token_id=tokenizer.convert_tokens_to_ids(
                        MAPPING_LANG_TO_KEY[args.target_language]
                    ),
                    max_new_tokens=output_len,
                    temperature=0.0,
                    top_p=1.0,
                    num_beams=5,
                    num_return_sequences=1,
                    do_sample=False,
                )
                outputs = tokenizer.batch_decode(
                    translated_tokens, skip_special_tokens=True
                )
            else:
                if args.number_of_demonstrations > 0:
                    batch_of_demonstrations = [
                        retriever.query(
                            sentence=sentence, k=args.number_of_demonstrations
                        )
                        for sentence in flat_list_of_batch_of_sentences
                    ]
                else:
                    batch_of_demonstrations = [
                        [] for _ in flat_list_of_batch_of_sentences
                    ]
                outputs = sampler.translate(
                    sentences=flat_list_of_batch_of_sentences,
                    demonstrations=batch_of_demonstrations,
                    **generation_kwargs,
                )
            assert len(outputs) == len(
                flat_list_of_batch_of_sentences
            ), f"Number of outputs ({len(outputs)}) does not match the number of sentences ({len(flat_list_of_batch_of_sentences)})"
            if args.use_nllb:
                # Scoring
                src_embs = text_embedder.predict(
                    flat_list_of_batch_of_sentences,
                    source_lang=MAPPING_LANG_TO_KEY[languages[i]],
                )
                mt_embs = text_embedder.predict(outputs, source_lang=MAPPING_LANG_TO_KEY[args.target_language])
                scores_tensor = blaser(src=src_embs, mt=mt_embs)
                scores = scores_tensor.reshape(-1).tolist()
            else:
                scores = [None] * len(flat_list_of_batch_of_sentences)
            current_index = 0
            for p in range(len(list_of_batch_of_sentences)):
                batch_of_sentences = list_of_batch_of_sentences[p]
                batch_of_scores = scores[
                    current_index : current_index + len(batch_of_sentences)
                ]
                translated_sentences = outputs[
                    current_index : current_index + len(batch_of_sentences)
                ]
                # Filter the sentences
                correct_language_indices = [
                    q
                    for q in range(len(translated_sentences))
                    if is_lang(translated_sentences[q], args.target_language)  # Right language
                    and len(translated_sentences[q]) >= 40  # Long enough
                    and translated_sentences[q].strip().endswith(".")  # Ends with a dot
                ]
                # save the translations
                if args.verbose:
                    for k in range(len(batch_of_sentences)):
                        print(
                            f"{current_index + k + 1}. T -> {translated_sentences[k]}\nS -> {batch_of_sentences[k]}\nscore -> {batch_of_scores[k]}\n"
                        )
                    print(
                        f"{len(correct_language_indices)}/{len(batch_of_sentences)} are in the correct language and have the right format."
                    )
                with open(
                    os.path.join(path_to_translations, filename), "a", encoding="utf-8"
                ) as fout:
                    dico = {
                        "sentence": batch_of_instructions[p],
                        "propositions": [
                            batch_of_sentences[q] for q in correct_language_indices
                        ],
                        "translations": [
                            translated_sentences[q] for q in correct_language_indices
                        ],
                        "scores": [
                            batch_of_scores[q] for q in correct_language_indices
                        ],
                    }
                    if batch_of_topics:
                        dico["topic"] = batch_of_topics[p]
                    fout.write(json.dumps(dico) + "\n")
                current_index += len(batch_of_sentences)

    print("END TRANSLATION")


def get_si_prompt(language: str, examples: List[str] = None):
    name = "passages"
    GENERATE_PROMPT = f"""
You are an helpful and precise polyglot assistant which is highly proficient in multiple languages including {language}.
""".strip()
    GENERATE_PROMPT += f"""
You are asked to come up with 20 {name}.

Here are the requirements:
1. The {name} should be in {language}.
2. The instructions should be from 1 to 3 sentences long. Imperative sentences and questions are permitted.
3. The {name} should be grammaticaly correct with respect to the rules of the {language} language.
4. Generate diverse {name}, that cover a wide range of topics about not necessarily related to {language}-speaking countries.
5. DO NOT INCLUDE THEIR ENGLISH TRANSLATION IN ANY FORM, WHETHER IN PARENTHESES OR OTHERWISE.

Complete the following list of {name} while adhering to the above requirements and maintaining the format of the initial instances. Write only the passages, without any additional content.

List of 20 {name}:
"""
    EXAMPLES = (
        "\n\n##\n\n".join([f"{i+1}. {example}" for i, example in enumerate(examples)])
        + "\n\n##"
    )
    prompt = GENERATE_PROMPT + "\n\n" + EXAMPLES
    return prompt


def self_instruct(args):
    rng = np.random.default_rng(args.seed)
    languages = args.languages

    dico_of_example_sentences = {}
    for language in languages:
        dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])["dev"]
        dico_of_example_sentences[language] = [
            example["sentence"] for example in dataset
        ]

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    template_key = args.template_key if args.template_key is not None else 11

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": "English",
        "tgt": languages[0],
        "template": get_template(key=template_key, src="English", tgt=languages[0]),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
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

    os.makedirs(args.output_dir, exist_ok=True)

    output_filenames = (
        args.output_filenames
        if args.output_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(output_filenames) == len(
        languages
    ), f"The number of output filenames ({len(output_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_machine_instructions = {language: [] for language in languages}
    number_of_instructions = 0

    for i, output_filename in enumerate(output_filenames):
        if os.path.exists(os.path.join(args.output_dir, output_filename)):
            with open(os.path.join(args.output_dir, output_filename), "r") as fin:
                for line in fin:
                    dico_of_machine_instructions[languages[i]].append(
                        json.loads(line)["text"]
                    )
                    number_of_instructions += 1

    # Tokenize
    tokenized_machine_instructions = {
        key: [scorer._tokenizer.tokenize(inst) for inst in value]
        for key, value in dico_of_machine_instructions.items()
    }
    # Progress bar
    progress_bar = tqdm(total=args.number_of_instructions)
    progress_bar.update(number_of_instructions)
    # Number of cycles without adding data
    repetition = 0
    while number_of_instructions < args.number_of_instructions or (
        args.number_of_instructions_per_language is not None
        and any(
            len(dico_of_machine_instructions[languages[i]])
            < args.number_of_instructions_per_language[i]
            for i in range(len(languages))
        )
    ):
        if repetition >= 50:
            break

        idx_language = rng.choice(len(languages), size=1).tolist()[0]
        language = languages[idx_language]
        prompts = []

        for _ in range(args.request_batch_size):
            # sample seed instructions
            seed_indices = rng.choice(
                a=np.arange(len(dico_of_example_sentences[language])),
                size=args.number_of_icl_demonstrations
                - args.number_of_icl_synthetic_demonstrations,
                replace=False,
            )
            # sample machine generated instructions
            if (
                len(dico_of_machine_instructions[language])
                >= args.number_of_icl_synthetic_demonstrations
            ):
                synthetic_indices = rng.choice(
                    a=np.arange(len(dico_of_machine_instructions[language])),
                    size=args.number_of_icl_synthetic_demonstrations,
                    replace=False,
                )
                prompt_instructions = [
                    dico_of_example_sentences[language][p] for p in seed_indices
                ] + [
                    dico_of_machine_instructions[language][p] for p in synthetic_indices
                ]
            else:
                synthetic_indices = rng.choice(
                    a=np.arange(len(dico_of_example_sentences[language])),
                    size=args.number_of_icl_synthetic_demonstrations,
                    replace=False,
                )
                prompt_instructions = [
                    dico_of_example_sentences[language][p] for p in seed_indices
                ] + [dico_of_example_sentences[language][p] for p in synthetic_indices]

            rng.shuffle(prompt_instructions)
            prompts.append(get_si_prompt(language, prompt_instructions))
            if args.verbose:
                pass
                # print(f"Prompt {len(prompts)}\n{prompts[-1]}\n###")
        answers = sampler.generate(
            [sampler.apply_chat_template(prompt) for prompt in prompts],
            **generation_kwargs,
        )
        new_instructions = []
        number_of_candidate_instructions = 0
        for answer in answers:
            answer = answer[0]
            answer = _stop_at_stop_token(
                # answer, [f"{args.number_of_icl_demonstrations + args.number_of_generations_per_step + 1}. "] + STOP_WORDS
                answer,
                STOP_WORDS,
            )
            if f"{args.number_of_icl_demonstrations + 1}. " in answer:
                # pattern = r"(\d+\. )"
                pattern = r"(\d+\. ?(?:\n)?)"
            else:
                pattern = r"(\d+\.\n)"
            splitted_answer = re.split(pattern, answer)
            candidates = []
            for j, element in enumerate(splitted_answer):
                if j == 0:
                    # everything before the first match
                    continue
                if j % 2 == 1:
                    # matches the iterator
                    continue
                candidate = element.replace("#", "").strip().split("\n\n")[0].strip()
                if len(candidate) <= 10:
                    continue
                print(f"{number_of_candidate_instructions + j//2}. {candidate}")
                candidates.append(candidate)
            number_of_candidate_instructions += len(candidates)
            new_instructions.append(candidates)

        kept_instructions = []
        number_of_kept_instructions = 0
        start_cleaning = time.time()
        for candidates in new_instructions:
            for candidate in candidates:
                candidate_tokens = scorer._tokenizer.tokenize(candidate)
                with Pool(4) as p:
                    rouge_scores = p.map(
                        partial(rouge_scorer._score_lcs, candidate_tokens),
                        tokenized_machine_instructions[language],
                    )
                rouge_scores = [score.fmeasure for score in rouge_scores]
                if len(rouge_scores) > 0 and max(rouge_scores) > args.threshold:
                    continue

                dico_of_machine_instructions[language].append(candidate)
                tokenized_machine_instructions[language].append(candidate_tokens)
                # Kept instructions
                number_of_kept_instructions += 1
                kept_instructions.append(candidate)
                progress_bar.update(1)
        print(
            f"We keep {number_of_kept_instructions}/{number_of_candidate_instructions} intructions, in {(time.time() - start_cleaning):.2f}s"
        )
        with open(
            os.path.join(args.output_dir, output_filenames[idx_language]),
            "a",
            encoding="utf-8",
        ) as fout:
            for instruction in kept_instructions:
                fout.write(json.dumps({"text": instruction}, ensure_ascii=False) + "\n")

        number_of_instructions += len(kept_instructions)
        if number_of_kept_instructions == 0:
            repetition += 1
        else:
            repetition = 0


def get_knn_prompt(language: str, examples: str = List[str]):
    HEADER_PROMPT = f"""
You are an helpful and precise polyglot assistant which is highly proficient in multiple languages including {language}.
""".strip()
    GENERATE_PROMPT = f"""
Position yourself as {language} news reporter, and craft a new, high-quality passage.
Keep the following in mind:
1. Relevance: Incorporate your previous analysis and make full use of this informative prior to ensure that the new passage is accurate and adheres to the {language} language rules.
2. Originality: The new passage should be distinguished to existing ones instead of naive imitation or transfer, so try your best in CREATIVITY.
3. Standalone: The new passage should be self-contained and not depend on prior passages.
4. Format: You should simply return the new passage without line breaks, nothing else.

Now, provide the new high-quality passage.
"""
    EXAMPLES = (
        "\n\nHere are a few passages you wrote.\n<Examples>\n"
        + "\n##\n".join(
            [f"{i+1}. {example}" for i, example in enumerate(examples[::-1])]
        )
        + "\n</Examples>\n"
    )
    return HEADER_PROMPT + EXAMPLES + GENERATE_PROMPT.strip()


def knn_instruct(args):
    languages = args.languages
    template_key = args.template_key if args.template_key is not None else 11

    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embedder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder",
        tokenizer="text_sonar_basic_encoder",
        # device=device,
    )

    dico_of_example_sentences = {}
    for language in languages:
        dataset = load_dataset("facebook/flores", MAPPING_LANG_TO_KEY[language])["dev"]
        dico_of_example_sentences[language] = (
            [example["sentence"] for example in dataset]
            if (args.max_samples is None or args.max_samples < -1)
            else [example["sentence"] for example in dataset][: args.max_samples]
        )

    arguments = {
        "model_name_or_path": args.model_name_or_path,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "src": "English",
        "tgt": languages[0],
        "template": get_template(key=template_key, src="English", tgt=languages[0]),
        "merge_prompt": "vanilla",
        "selection_method": "greedy",
        "method_translate": "vanilla",
        "nllb_name_or_path": None,
        "method_divide": None,
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

    print(f"Sampler set up!")
    os.makedirs(args.output_dir, exist_ok=True)

    output_filenames = (
        args.output_filenames
        if args.output_filenames is not None
        else [f"{language}.jsonl" for language in languages]
    )

    assert len(output_filenames) == len(
        languages
    ), f"The number of output filenames ({len(output_filenames)}) should match the number of languages ({len(languages)})."
    dico_of_machine_instructions = {language: [] for language in languages}
    for i, output_filename in enumerate(output_filenames):
        if os.path.exists(os.path.join(args.output_dir, output_filename)):
            with open(os.path.join(args.output_dir, output_filename), "r") as fin:
                for line in fin:
                    dico_of_machine_instructions[languages[i]].append(
                        json.loads(line)["text"]
                    )

    for language in languages:
        print(f"Processing {language}.")
        # Load the seed embeddings
        if not os.path.exists(os.path.join(args.output_dir, f"{language}_0.bin")):
            X_emb = text_embedder.predict(
                dico_of_example_sentences[language],
                source_lang=MAPPING_LANG_TO_KEY[language],
                batch_size=args.request_batch_size,
                progress_bar=True,
            )
            X = (
                X_emb.detach().numpy()
                if device == "cpu"
                else X_emb.cpu().detach().numpy()
            )
            X.tofile(os.path.join(args.output_dir, f"{language}_0.bin"))
        # Find the round to start to
        start_round = 0
        while start_round <= args.number_of_rounds and os.path.exists(
            os.path.join(args.output_dir, f"{language}_{start_round}.bin")
        ):
            start_round += 1

        if start_round > args.number_of_rounds:
            # The generation pipeline is over
            # exit()
            continue
        # Concatenate
        seed_instructions = dico_of_example_sentences[language]
        # Load the seed embeddings
        X = np.fromfile(
            os.path.join(args.output_dir, f"{language}_0.bin"),
            dtype=np.float32,
            count=-1,
        ).reshape(-1, 1024)
        for r in range(1, start_round):
            seed_instructions_r = []
            with open(
                os.path.join(args.output_dir, f"{language}_{r}.jsonl"), "r"
            ) as fin:
                for line in fin:
                    seed_instructions_r.append(json.loads(line)["text"])
            X_r = np.fromfile(
                os.path.join(args.output_dir, f"{language}_{r}.bin"),
                dtype=np.float32,
                count=-1,
            ).reshape(-1, 1024)
            seed_instructions.extend(seed_instructions_r)
            X = np.concatenate((X, X_r), axis=0)
        print(f"A ({len(seed_instructions)}) should be equal to B ({X.shape[0]})")
        for round in range(start_round, args.number_of_rounds + 1):
            print(f"X.shape: {X.shape}")
            start = 0
            current_round_instructions = []
            if os.path.exists(
                os.path.join(args.output_dir, f"{language}_{round}.jsonl")
            ):
                with open(
                    os.path.join(args.output_dir, f"{language}_{round}.jsonl"), "r"
                ) as fin:
                    for line in fin:
                        current_round_instructions.append(json.loads(line)["text"])
                        start += 1
            if start >= len(seed_instructions):
                continue
            print(f"Starting from index {start}")
            if start > 0:
                try:
                    Z = np.fromfile(
                        os.path.join(args.output_dir, f"{language}_temp.bin"),
                        dtype=np.float32,
                        count=-1,
                    ).reshape(-1, 1024)
                    print(f"Start({start}) should be equal to {Z.shape[0]}.")
                except Exception as ex:
                    print(f"Exception: {ex}")
            # Progress bar
            progress_bar = tqdm(total=len(seed_instructions))
            progress_bar.update(start)
            for i in range(start, len(seed_instructions), args.request_batch_size):
                from sklearn.metrics import pairwise_distances

                # Similarity matrix
                D = 1 - pairwise_distances(
                    X[i : min(i + args.request_batch_size, len(seed_instructions)), :],
                    X,
                    metric="cosine",
                )
                D = np.argsort(D, axis=-1)[
                    :, -(args.number_of_icl_demonstrations + 1) :
                ]
                print(f"D.shape: {D.shape}")
                # assert all([D[j][-1] == i + j for j in range(len(D))])
                print(
                    f"Good ratio: {sum([D[j][-1] == i + j for j in range(len(D))]) / len(D)}"
                )
                list_of_indices = [D[j][:-1].tolist() for j in range(len(D))]
                print(f"list_of_indices: {list_of_indices}")
                list_of_instructions = [
                    [seed_instructions[j] for j in indices]
                    for indices in list_of_indices
                ]
                # print(f"list_of_instructions: {list_of_instructions}")
                # From the least similar to the most similar
                prompts = [
                    get_knn_prompt(language, instructions + [seed_instructions[i + j]])
                    for j, instructions in enumerate(list_of_instructions)
                ]
                if args.verbose:
                    if i == 0:
                        print(f"{'-'*30}\nPROMPT\n{'-'*30}\n{prompts[0]}\n{'-'*30}")

                answers = sampler.generate(
                    [sampler.apply_chat_template(prompt) for prompt in prompts],
                    **generation_kwargs,
                )
                new_instructions = []
                for j, answer in enumerate(answers):
                    answer = answer[0].strip()
                    answer = _stop_at_stop_token(
                        answer,
                        [f"{args.number_of_icl_demonstrations + 2}. "] + STOP_WORDS,
                    )
                    answer = answer.split("\n")[0].strip()
                    if answer.strip() == "":
                        print(f"Empty generation.")
                        # continue
                    print(f"{j+1} -> {answer}")
                    progress_bar.update(1)
                    new_instructions.append(answer)
                print(f"Adding {len(new_instructions)} new instructions.")
                # Update current instructions
                current_round_instructions.extend(new_instructions)
                Y_emb = text_embedder.predict(
                    new_instructions, source_lang=MAPPING_LANG_TO_KEY[language]
                )
                Y = (
                    Y_emb.detach().numpy()
                    if device == "cpu"
                    else Y_emb.cpu().detach().numpy()
                )
                # Update embeddings
                try:
                    # Load the temporary embeddings
                    temp = np.fromfile(
                        os.path.join(args.output_dir, f"{language}_temp.bin"),
                        dtype=np.float32,
                        count=-1,
                    ).reshape(-1, 1024)
                    print(f"temp.shape before: {temp.shape}")
                    temp = np.concatenate((temp, Y), axis=0)
                    print(f"temp.shape after: {temp.shape}")
                    temp.tofile(os.path.join(args.output_dir, f"{language}_temp.bin"))
                except Exception as e:
                    print(f"ERROR: {e}")
                    temp = Y
                    print(f"temp.shape now: {temp.shape}")
                    temp.tofile(os.path.join(args.output_dir, f"{language}_temp.bin"))
                # Save new instructions
                with open(
                    os.path.join(args.output_dir, f"{language}_{round}.jsonl"), "a"
                ) as fout:
                    for instruction in new_instructions:
                        fout.write(
                            json.dumps({"text": instruction}, ensure_ascii=False) + "\n"
                        )
            print(f"END OF ROUND {i+1}")
            # Add all synthesized conversations to Seeds
            seed_instructions.extend(current_round_instructions)
            try:
                X = np.concatenate((X, temp), axis=0)
            except Exception as ex:
                print(f"Exception: {ex}")
            assert (
                len(seed_instructions) == X.shape[0]
            ), f"The number of seed instructions ({len(seed_instructions)}) does not match with the embedding shape {X.shape}"
            # Save the temporary embeddings to be final
            temp.tofile(os.path.join(args.output_dir, f"{language}_{round}.bin"))
            # Delete the temporary embeddings
            os.system(f"rm -r {os.path.join(args.output_dir, f'{language}_temp.bin')}")
    print(f"END 1")
    for language in languages:
        r = 0
        instructions_r = []
        while os.path.exists(os.path.join(args.output_dir, f"{language}_{r}.jsonl")):
            with open(
                os.path.join(args.output_dir, f"{language}_{round}.jsonl"), "r"
            ) as fin:
                for line in fin:
                    instructions_r.append(json.loads(line)["text"])
            r += 1
        with open(os.path.join(args.output_dir, f"{language}.jsonl"), "a") as fout:
            for instruction in instructions_r:
                fout.write(json.dumps({"text": instruction}, ensure_ascii=False) + "\n")
    print("END")


if __name__ == "__main__":
    args = parse_args()
    if args.data_generation_pipeline == "topxgen":
        main(args)
    elif args.data_generation_pipeline == "self-instruct":
        self_instruct(args)
    elif args.data_generation_pipeline == "knn-instruct":
        knn_instruct(args)
    else:
        raise ValueError(f"Unsupported data generation pipeline ({args.data_generation_pipeline})")
