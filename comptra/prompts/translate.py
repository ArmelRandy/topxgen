from typing import List, Tuple
from comptra.prompts.templates import Template, get_template


def get_translate_prompt(
    sentence: str,
    src: str = "English",
    tgt: str = "French",
    demonstrations: List[Tuple[str, str]] = [],
    template: Template = None,
    ift: bool = True,
) -> str:
    """
    Build a few-shot or zero-shot Machine Translation prompt.
    Arguments
    ---------
        - sentence : str,
            Sequence that is to be translated
        - src : str,
            Source language (e.g. English)
        - tgt : str,
            Target language (e.g. French)
        - demonstrations : List[(str, str)], default = []
            List of pairs (sentence in source language, translation in target language) used as demonstrations for ICL
        - template : Template, default = None
            Template used for MT
        - ift: bool
            if the model is instruction fine-tuned.
    Returns
    -------
        - str :
            Few-shot prompt for Machine Translation from `src` to `tgt` using the examples in `demonstrations`
    """

    if not ift:
        template = template if template else get_template(11, src, tgt)
        prompt = template.get_prompt(demonstrations, sentence)
    else:
        prompt = ""
        if len(demonstrations) != 0:
            prompt += "Given the following sentence-translation pairs written by a professional translator:\n\n<Demonstrations>\n"
            if template:
                for i, (source, target) in enumerate(demonstrations):
                    prompt += (
                        f"{i+1}. {template.prefix}{source}{template.middle}{target}\n\n"
                    )
                prompt = prompt.strip() + "\n</Demonstrations>\n\n"
            else:
                for i, (source, target) in enumerate(demonstrations):
                    prompt += f"{i+1}. {src} sentence\n{source}\n{tgt} translation\n{target}\n\n"
                prompt = prompt.strip() + "\n</Demonstrations>\n\n"
        prompt += f"Please write a high-quality {tgt} translation of the following {src} sentence\n\n{sentence}\n\n"
        if len(demonstrations) != 0:
            prompt += "Please make sure to consider the above information and provide only the translation, nothing more."
        else:
            prompt += "Please provide only the translation, nothing more."
    return prompt


import spacy

nlp = spacy.load("en_core_web_sm")


def get_cot_prompt(
    sentence: str,
    src: str,
    tgt: str,
    demonstrations: List[Tuple[str, str]],
    template: Template,
) -> str:
    prompt = ""
    if len(demonstrations) != 0:
        prompt += "Given the following sentence-translation pairs written by a professional translator:\n\n<Demonstrations>\n"
        if template:
            for i, (source, target) in enumerate(demonstrations):
                prompt += (
                    f"{i+1}. {template.prefix}{source}{template.middle}{target}\n\n"
                )
            prompt = prompt.strip() + "\n</Demonstrations>\n\n"
        else:
            for i, (source, target) in enumerate(demonstrations):
                prompt += (
                    f"{i+1}. {src} sentence\n{source}\n{tgt} translation\n{target}\n\n"
                )
            prompt = prompt.strip() + "\n</Demonstrations>\n\n"

    prompt += f"Please write a high-quality {tgt} translation of the following {src} sentence\n\n{sentence}"
    # Structural Information
    """
    prompt += "\n\nMake sure to consider the following information:\n"
    prompt += "- The provided sentences share some words with the sentence to translate, you can use them for inspiration in terms of vocabulary, script, style and phrasing.\n"
    prompt += "- Each language has an ordering e.g. English is a SVO (Subject-Verb-Object) while Japanese is SOV.\n"
    prompt += "- The meaning of a word depends on its context and some languages require to add a grammatical gender to objects.\n"
    dico = {
        "GPE": "Geopolitical entities, i.e. countries, cities, states.",
        "ORG": "Companies, agencies, institutions.",
        "MONEY": "Monetary values, including unit.",
        "PERSON": "Persons.",
        "DATE": "Dates or times.",
    }
    doc = nlp(sentence)
    count = sum([1 for ent in doc.ents if ent.label_ in dico])
    if count > 0:
        prompt += "Moreover,\n"
        for category in dico:
            L = []
            for ent in doc.ents:
                if category in ent.label_:
                    L.append(ent.text)
            if len(L) == 0:
                pass
            elif len(L) == 1:
                prompt += f"- {L[0]} belongs to {dico[category]}\n"
            else:
                prompt += (
                    f"- {', '.join(L[:-1])} and {L[-1]} belong to {dico[category]}\n"
                )
    """
    prompt = prompt.strip() + "\n\nLet's think step by step."
    return prompt
