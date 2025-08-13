import os
import json
from comptra.prompts.templates import get_template
from comptra.sampler import *

path = os.path.join(os.path.dirname(__file__), "data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B")

print(f"{os.path.exists(path)}")

LANGUAGES = [
    "Basque",
    "Hausa",
    "Igbo",
    "Kinyarwanda",
    "Nepali",
    "Somali",
    "Sundanese",
    "Swahili",
    "Urdu",
    "Xhosa"
]

model_name_or_path = "google/gemma-3-27b-it"

arguments = {
    "model_name_or_path": model_name_or_path,
    "tokenizer_name_or_path": model_name_or_path,
    "src": "French",
    "tgt": "English",
    "template": get_template(key=template_key, src="French", tgt="English"),
    "merge_prompt": "vanilla",
    "selection_method": "greedy",
    "method_translate": "vanilla",
    "nllb_name_or_path": None,
    "method_divide": None,
}

generation_kwargs = {
    "max_new_tokens": 100,
    "temperature": 0.0,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "num_return_sequences": 1,
    "num_beams": 1,
    "do_sample": False,
    "request_batch_size": 128,
    "verbose": True,
}

sampler = vLLMSampler(**arguments)

def get_prompt(paragraph, lang, topic):
    prompt = f"Does the following {lang} paragraph\n\n<paragraph>\n{paragraph}\n</paragraph>\n"
    prompt += f"discuss the following topic\n\n<topic>\n{topic}\n</topic>\n\n"
    prompt += "Think about it step by step and answer with Yes or No."
    prompt += "\n\nHere is the output format\n\n<reasoning>\n...\n<reasoning>\n<answer>\nYes or No\n</answer>"
    prompt += "\n\nMake sure to follow the output format and DO NOT WRITE ANYTHING AFTER THE ANSWER."
    return prompt

for language in LANGUAGES:
    current = os.path.join(
        path, f"{language}_translate.jsonl"
    )
    topics = []
    paragraphs = []
    list_of_sentences = []
    with open(current, "r") as fin:
        for line in fin:
            dico = json.loads(line)
            paragraphs.append(dico["sentence"])
            topics.append(dico["topic"])
            list_of_sentences.append(dico["translations"])
    
    output_filename = f"{language}_statistics.jsonl"
    start = 0
    if os.path.exists(os.path.join(path, output_filename)):
        with open(
            os.path.join(path, output_filename),
            "r"
        ) as fin:
            for line in fin:
                start += 1
    
    batch_size = 128
    for i in range(start, len(topics), batch_size):
        source = [
            get_prompt(
                paragraphs[j], language, topics[j]
            )
            for j in range(i, i + batch_size)
        ]
        target = [
            get_prompt(
                " ".join(list_of_sentences[j]), "English", topics[j]
            )
            for j in range(i, i + batch_size)            
        ]
        if i == start:
            print(f"{'-'*75}\nFirst source.\n{'-'*75}\n{source[0]}\n{'-'*75}")
            print(f"{'-'*75}\nFirst target.\n{'-'*75}\n{target[0]}\n{'-'*75}")
        source_answers = sampler.generate(
            source, **generation_kwargs
        )
        target_answers = sampler.generate(
            target, **generation_kwargs
        )
        with open(os.path.join(path, output_filename), "a") as fout:
            for s, t in zip(source_answers, target_answers):
                fout.write(
                    json.dumps(
                        {
                            "source": s[0],
                            "target": t[0]
                        }
                    ) + "\n"
                )