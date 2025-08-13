import tqdm
import torch
from typing import List
from accelerate.utils import gather_object
from transformers import StoppingCriteria
from torch.utils.data import IterableDataset

import numpy as np

class EndOfFunctionCriteria(StoppingCriteria):
    """Custom `StoppingCriteria` which checks if all generated functions in the batch are completed."""

    def __init__(self, start_length, eof_strings, tokenizer):
        self.start_length = start_length
        self.eof_strings = eof_strings
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        """Returns true if all generated sequences contain any of the end-of-function strings."""
        decoded_generations = self.tokenizer.batch_decode(
            input_ids[:, self.start_length :]
        )
        done = []
        for decoded_generation in decoded_generations:
            done.append(
                any(
                    [
                        stop_string in decoded_generation
                        for stop_string in self.eof_strings
                    ]
                )
            )
        return all(done)


class TokenizedDataset(IterableDataset):
    """Tokenize and preprocess the dataset, where the dataset is a list of instructions (str)"""

    def __init__(self, tokenizer, dataset):
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.outputs = self.tokenizer(self.dataset, padding=True, return_tensors="pt")

    def __iter__(self):
        for i in range(len(self.dataset)):
            yield {
                "input_ids": self.outputs.input_ids[i],
                "attention_mask": self.outputs.attention_mask[i],
                "index_prompt": torch.tensor(i, dtype=torch.int32),
            }


def hf_generate(
    accelerator,
    model,
    tokenizer,
    prompts,
    max_new_tokens,
    temperature,
    top_p,
    stop_words,
    num_beams,
    repetition_penalty,
    num_return_sequences,
    do_sample,
    forced_bos_token_id=None,
    batch_size=1,
    verbose=False,
):
    """
    Inspired by
    https://github.com/geronimi73/accelerate_tricks/blob/main/inference_batched.py
    """
    accelerator.free_memory()
    with accelerator.split_between_processes(prompts) as prompts_:
        prompt_batches = [
            prompts_[i : i + batch_size] for i in range(0, len(prompts_), batch_size)
        ]
        results = []
        for prompt_batch in tqdm.tqdm(prompt_batches):
            tokenized_batch = tokenizer(
                prompt_batch, padding=True, return_tensors="pt"
            ).to("cuda")
            tokenized_outputs = model.generate(
                **tokenized_batch,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                num_return_sequences=num_return_sequences,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=do_sample,
                forced_bos_token_id=forced_bos_token_id,
                stop_strings=stop_words,
                tokenizer=tokenizer,
            )
            tokenized_outputs = [
                tokenized_outputs[i][
                    len(tokenized_batch["input_ids"][i // num_return_sequences]) :
                ]
                for i in range(len(tokenized_outputs))
            ]
            for i in range(0, len(tokenized_outputs), num_return_sequences):
                if verbose:
                    print(
                        f"---<>---\n{prompt_batch[i // num_return_sequences]}\n{tokenizer.decode(tokenized_outputs[i], skip_special_tokens=True)}\n---<>---"
                    )
                results.append(
                    {
                        "prompt": prompt_batch[i // num_return_sequences],
                        "answer": tokenizer.batch_decode(
                            tokenized_outputs[i : i + num_return_sequences],
                            skip_special_tokens=True,
                        ),
                    }
                )
    results_gathered = gather_object(results)
    if verbose:
        for prompt, result in zip(prompts, results_gathered):
            out = f"{prompt}\n"
            for j in range(len(result["answer"])):
                out += f"{j+1}. {result['answer'][j]}\n"
            print(f"===\n{out}\n===")
    return results_gathered

#"""
from comet import load_from_checkpoint, download_model

model_path = download_model("Unbabel/wmt20-comet-qe-da")
qe = load_from_checkpoint(model_path)
#"""

def quality_estimation(sources: List[str], predictions: List[str], batch_size: int = 8):
    data = [
        {"src": source, "mt": prediction}
        for source, prediction in zip(sources, predictions)
    ]
    with torch.no_grad():
        scores = qe.predict(data, batch_size=batch_size, gpus=0).scores
    scores = np.array(scores)
    return scores


import fasttext
from comptra.languages import MAPPING_LANG_TO_KEY
from huggingface_hub import hf_hub_download

model_name_or_path = hf_hub_download(
    repo_id="facebook/fasttext-language-identification", filename="model.bin"
)
language_identifier = fasttext.load_model(model_name_or_path)


def is_lang(sentence: str, lang: str):
    """
    Takes as input a sentence and a language and output whether the sentence is written in that language.
    Arguments
    ---------
        - sentence : str,
            A given sentence
        - lang :
            A language (e.g. English, French, German etc.)
    """
    label, p = language_identifier.predict(sentence.strip().split("\n")[0])
    # print(f"probability: {p[0]}")
    label = label[0]
    return MAPPING_LANG_TO_KEY[lang] in label


#"""
from sonar.models.blaser.loader import load_blaser_model
from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline


blaser_qe = load_blaser_model("blaser_2_0_qe").eval()
text_embedder = TextToEmbeddingModelPipeline(
    encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
)

def get_blaser_score(x, y, src, tgt):
    src_embs = text_embedder.predict([x], source_lang=MAPPING_LANG_TO_KEY[src])
    ref_embs = text_embedder.predict([y], source_lang=MAPPING_LANG_TO_KEY[tgt])
    blaser_score = blaser_qe(src=src_embs, mt=ref_embs).item()
    return blaser_score
#"""

from sacrebleu.metrics import BLEU
bleu = BLEU(tokenize="flores200")

def get_best_sentence(
    target_translations: List[str],
    src: str = None,
    tgt: str = None,
    source_sentence: List[str] = None,
    strategy: str = "greedy",
    verbose: bool = False,
):
    """
    Takes as input a list of translations, compute a score for each of them and return the sentence with the highest score
    Arguments
    ---------
        - target_translatoins : List[str],
            translations we would like to score.
        - sources_sentence : str, default = None
            sentence whose candidate translations are provided.
        - strategy: str
            How to score the translations.
            score(s_i) = \sum_{j=1, j!= i}^{N}sim(s_i, s_j)
    """
    if strategy == "greedy" or len(target_translations) == 1:
        return target_translations[0]
    elif strategy == "bleu":
        assert (
            source_sentence is not None
        ), "The `bleu` strategy requires to specify the source sentence"
        idx = -1
        max_score = float("-inf")
        for i in range(len(target_translations)):
            if target_translations[i].strip() == "":
                # Ignore empty translations
                continue
            c = 0
            for j in range(len(target_translations)):
                if j != i:
                    b = bleu.corpus_score(
                        [target_translations[j]], [[target_translations[i]]]
                    ).score
                    c += b
            if c > max_score:
                max_score = c
                idx = i
        if idx == -1:
            warnings.warn("All the sentences are empty")
            return ""
    elif strategy == "blaser":
        assert all(
            [col is not None for col in [src, tgt, source_sentence]]
        ), "The `blaser` strategy requires to specify the source language (src)\
            , the target language (tgt) and the source sentence (source_sentence)."
        scores = [
            is_lang(target_translation, tgt)
            * get_blaser_score(source_sentence, target_translation, src, tgt)
            for target_translation in target_translations
        ]
        idx = np.argmax(scores)
    elif strategy == "comet-qe":
        assert (
            source_sentence is not None
        ), "The `bleu` strategy requires to specify the source sentence"
        scores = quality_estimation(
            sources=[source_sentence] * len(target_translations),
            predictions=target_translations,
        )
        idx = np.argmax(scores)
    else:
        raise ValueError(f"Unsupported ensembling strategy {strategy}.")
    if verbose:
        prompt = "===\n"
        if source_sentence:
            prompt += f"Sentence\n{source_sentence}\n"
        prompt = "The best translation between the following\n"
        for i in range(len(target_translations)):
            prompt += f"{i+1}. {target_translations[i]}\n"
        prompt += f"Is\n{target_translations[idx]}"
        prompt += "\n==="
        print(prompt)
    return target_translations[idx]


from collections import Counter


def get_bigrams(sentence):
    words = sentence.split()
    bigrams = [(words[i], words[i + 1]) for i in range(len(words) - 1)]
    return bigrams


def count_bigrams(bigrams):
    return Counter(bigrams)


# https://github.com/bigcode-project/bigcode-evaluation-harness/blob/main/bigcode_eval/base.py#L83
def _stop_at_stop_token(decoded_string: str, stop_tokens: List[str]):
    """
    Produces the prefix of decoded_string that ends at the first occurrence of
    a stop_token.
    WARNING: the decoded_string *must not* include the prompt, which may have stop tokens
    itself.
    """
    min_stop_index = len(decoded_string)
    for stop_token in stop_tokens:
        stop_index = decoded_string.find(stop_token)
        if stop_index != -1 and stop_index < min_stop_index:
            min_stop_index = stop_index
    return decoded_string[:min_stop_index].rstrip()


def remove_repeating_bigram(sentence: str, repetition_threshold: int = 8):
    """
    Takes as input a sentence (str) and remove the trailing bigram if it exists
    """
    if len(sentence.strip()) == 0:
        return sentence
    try:
        bigrams = count_bigrams(get_bigrams(sentence))
        bigram, highest_frequency = bigrams.most_common()[0]
        if highest_frequency >= repetition_threshold:
            # There is a bigram that keep repeating itself in the sentence
            bigram_str = " ".join(bigram)
            print(f"The bigram '{bigram_str}' keep repeating itself in '{sentence}'.")
            return sentence[: sentence.find(bigram_str) + len(bigram_str)]
        else:
            return sentence
    except:
        return sentence

def lcs(a, b):
    N = len(a)
    M = len(b)
    if N != 0 or M != 0:
        return 0
    dp = [[0] * M for _ in range(N)]
    for j in range(M):
        if a[0] == b[j]:
            for k in range(j, M):
                dp[0][k] = 1
            break
    for i in range(N):
        if a[i] == b[0]:
            for k in range(i, N):
                dp[k][0] = 1
            break

    for i in range(1, N):
        for j in range(1, M):
            if a[i] == b[j]:
                dp[i][j] = 1 + dp[i - 1][j - 1]
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[N - 1][M - 1]

if __name__ == "__main__":
    print(
        get_best_sentence(
            target_translations = [
                "I want to be fed your pancreas.",
                "I want to eat your pancreas.",
                "I love eating pancreas."
            ],
            source_sentence="Je veux manger ton pancreas.",
            src = "French",
            tgt = "English",
            strategy = "bleu",
            verbose = True
        )
    )