# Template
from dataclasses import dataclass


@dataclass
class Template:
    """
    We are interested in the few-/zero-shot setup in order to perform a task such as mapping an
    input x to an output y. In machine translation for example, x is a sentence written in a source
    language (say english) and y is the corresponding sentence in the language of interest (say french)

    For this specific example, an example template could be
    English: {x} French: {y}

    Believe that a template is defined by 3 components
    - The `prefix` : That is, everything that comes before the input (i.e. x)
    - The `middle` : Everything that comes between the input (i.e. x) and the output (i.e. y)
    - The `suffix`: That is, the part after the output y_{k} and the next input x_{k+1} in k-shot learning
    """

    header: str = ""
    prefix: str = "[src]: "
    middle: str = "\n[tgt]: "
    suffix: str = "\n\n"

    def get_prompt(self, demonstrations, example, start="", end=""):
        """
        Takes as input a list of demonstrations (i.e. few-shot examples) and an input in order
        to build the prompt to be fed to a LLM.
        Example :

        demonstrations  = [
            (
                "What is the capital of Russia?",
                "The capital of Russia is Moscow"
            ),
            (
                "What is the capital of France?",
                "The capital of France is Paris."
            )
        ]
        example = "What is the capital of Japan?"

        """
        prompt = self.header
        if demonstrations:
            for x, y in demonstrations:
                prompt += f"{self.prefix}"
                prompt += f"{start}{x}{end}"
                prompt += f"{self.middle}"
                prompt += f"{start}{y}{end}"
                prompt += f"{self.suffix}"
        if example is not None:
            prompt += f"{self.prefix}"
            prompt += f"{start}{example}{end}"
            prompt += f"{self.middle}"
        
        if prompt.endswith(": "):
            prompt = prompt[:-1]
        
        return prompt

    def copy(self):
        return Template(
            header=self.header,
            prefix=self.prefix,
            middle=self.middle,
            suffix=self.suffix,
        )

    def update_src(self, old_src, new_src):
        return Template(
            header=self.header.replace(old_src, new_src),
            prefix=self.prefix.replace(old_src, new_src),
            middle=self.middle.replace(old_src, new_src),
            suffix=self.suffix,
        )
    
    def update_tgt(self, old_tgt, new_tgt):
        return Template(
            header=self.header.replace(old_tgt, new_tgt),
            prefix=self.prefix.replace(old_tgt, new_tgt),
            middle=self.middle.replace(old_tgt, new_tgt),
            suffix=self.suffix,
        )

MAPPING_LANG_TO_TRANSLATION = {
    "English": "English",
    "French": "Français",
    "German": "Deutsch",
    "Swahili": "Kiswahili",
    "Wolof": "Wolof",
    "Hindi": "हिन्दी",
    "Spanish": "Español",
    "Japanese": "日本語",
}

LEFT = {
    "English": "English sentence",
    "French": "Phrase en français",
    "German": "Satz auf Deutsch",
    "Swahili": "Sentensi ya Kiswahili",
    "Wolof": "Mbindum wolof",
    "Hindi": "",
    "Spanish": "",
    "Japanese": "",
}

RIGHT = {
    "English": "French translation",
    "French": "Traduction en français",
    "German": "Deutsche Übersetzung",
    "Swahili": "Tafsiri ya Kiswahili",
    "Wolof": "Mbinde buñu sirri si wolof",
    "Hindi": "",
    "Spanish": "",
    "Japanese": "",
}


def get_template(key: int, src: str, tgt: str) -> Template:
    header = ""
    prefix, middle, suffix = None, None, "\n\n"
    if key == 1:
        prefix = "Given the following source text: "
        middle = f", a good {tgt} translation is: "
    elif key == 2:
        prefix = f"Given the following source text in {src}: "
        middle = f", a good {tgt} translation is: "
    elif key == 3:
        prefix = "If the original version says "
        middle = f"then the {tgt} version should say: "
    elif key == 4:
        prefix = f"What is the {tgt} translation of the sentence: "
        middle = "?\n"
    elif key == 5:
        prefix = ""
        middle = f"= {tgt}: "
    elif key == 6:
        prefix = f"{src}: "
        middle = f"= {tgt}: "
    elif key == 7:
        prefix = f"{src}\n"
        middle = f"\ntranslates into\n{tgt}\n"
        suffix = "\n###\n"
    elif key == 8:
        prefix = f"{MAPPING_LANG_TO_TRANSLATION[src]}\n"
        middle = f"\ntranslates into\n{MAPPING_LANG_TO_TRANSLATION[tgt]}\n"
        suffix = "\n###\n"
    elif key == 9:
        prefix = f"{src}: "
        middle = f"\n{tgt}: "
        suffix = "\n###\n"
    elif key == 10:
        prefix = f"{MAPPING_LANG_TO_TRANSLATION[src]}: "
        middle = f"\n{MAPPING_LANG_TO_TRANSLATION[tgt]}: "
        suffix = "\n###\n"
    elif key == 11:
        prefix = f"{src} sentence\n"
        middle = f"\n{tgt} translation\n"
        suffix = "\n###\n"
    elif key == 12:
        prefix = f"{LEFT[src]}\n"
        middle = f"\n{RIGHT[tgt]}\n"
        suffix = "\n###\n"
    elif key == 13:
        header = ""
        prefix = f"Write a {tgt} translation of the following {src} sentence.\n"
        middle = "\n"
        suffix = "\n###\n"
    elif key == 14:
        header = ""
        prefix = f"Translate this from {src} to {tgt}:\n{src}: "
        middle = f"\n{tgt}: "
        suffix = "\n###\n"
    elif key == 15:
        # LLaMAX-Alpaca
        header = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n"
        prefix = f"###Instruction:\nTranslate the following sentences from {src} to {tgt}\n###Input:\n"
        middle = f"\n###Response:"
    elif key == 16:
        header = ""
        prefix = f"Translate to {tgt}: "
        middle = ""
        suffix = "\n###\n"
    elif key == 17:
        header = ""
        prefix = f"Translate this from {src} to {tgt}:\n{src}: "
        middle = f"\n{tgt}: <think>"
        suffix = "\n###\n"
    else:
        raise KeyError(
            f"The key {key} does not describe one of the ICL format that we support!"
        )

    return Template(header=header, prefix=prefix, middle=middle, suffix=suffix)


import spacy

nlp = spacy.load("en_core_web_sm")
from typing import List, Tuple


def get_linguistic_prompt(
    example: str,
    demonstrations: List[Tuple[str, str]],
    src: str,
    tgt: str,
    feature: str,  # in ["pos", "morph", "dep"]
    ift: bool = False
):
    prompt = f"Use the following example sentence-translation pairs written by a professional translator to translate a new sentence given below."
    if ift:
        prompt += "\nWrite the translation first (without anything before) and then explain your reasoning."
    prompt += "\n\n<Examples>\n"
    for i, (source, target) in enumerate(demonstrations):
        doc = nlp(source)
        prompt += f"{i+1}. {src} sentence\n{source}\n"
        if feature == "pos":
            pos = [
                f"{token.text} is {'an' if any([token.pos_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {token.pos_}"
                for token in doc
            ]
            pos_information = f"{', '.join(pos[:-1])} and {pos[-1]}."
            prompt += "\nPart-of-Speech tags (Universal Dependencies tagset)\n"
            prompt += f"{pos_information}\n\n"
        elif feature == "morph":
            pos = [
                f"{token.text} is {'an' if any([token.pos_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {token.pos_} (in particular {token.morph})"
                if token.morph
                else f"{token.text} is {'an' if any([token.pos_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {token.pos_}"
                for token in doc
            ]
            prompt += (
                "\nFine-grained part-of-speech tags (Universal Dependencies tagset)\n"
            )
            pos_information = f"{', '.join(pos[:-1])} and {pos[-1]}."
            prompt += f"{pos_information}\n\n"
        elif feature == "dep":
            tuples = []
            for token in doc:
                if token.is_punct:
                    continue
                children = [
                    element for element in token.children if not element.is_punct
                ]
                if len(children) == 0:
                    continue
                a = f"{token.dep_} ({token.text})"
                b = "  ".join([f"{child.dep_} ({child.text})" for child in children])
                tuples.append(f"{a} -> {b}")
            prompt += "\nTyped dependency structure\n"
            prompt += "We have the following dependencies:\n"
            prompt += "\n".join(tuples) + "\n\n"
            # prompt += "\n".join(tuples[:-1]) + f"\nand\n{tuples[-1]}.\n\n"
        elif feature == "ner":
            ner = [
                f"{ent.text} is {'an' if any([ent.label_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {ent.label_}"
                for ent in doc.ents
            ]
            if len(ner) == 0:
                pass
            else:
                if len(ner) == 1:
                    ner_information = ner[0]
                else:
                    ner_information = f"{', '.join(ner[:-1])} and {ner[-1]}"
                prompt += "\nNamed Entities tags\n"
                prompt += f"{ner_information}\n\n"
        elif feature == "none":
            prompt + "\n"
        else:
            raise KeyError(
                f"{feature} is not supported. Choose one of ['pos', 'morph', 'dep', 'ner', 'none']"
            )
        prompt += f"{tgt} translation\n{target}\n\n"

    prompt = prompt.strip() + "\n</Examples>\n\n"
    prompt += f"{src} sentence\n{example}\n"
    doc = nlp(example)
    if feature == "pos":
        pos = [
            f"{token.text} is {'an' if any([token.pos_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {token.pos_}"
            for token in doc
        ]
        pos_information = f"{', '.join(pos[:-1])} and {pos[-1]}."
        prompt += "\nPart-of-Speech tags (Universal Dependencies tagset)\n"
        prompt += f"{pos_information}\n\n"
    elif feature == "morph":
        pos = [
            f"{token.text} is {'an' if any([token.pos_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {token.pos_} (in particular {token.morph})"
            if token.morph
            else f"{token.text} is {'an' if any([token.pos_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {token.pos_}"
            for token in doc
        ]
        prompt += "\nFine-grained part-of-speech tags (Universal Dependencies tagset)\n"
        pos_information = f"{', '.join(pos[:-1])} and {pos[-1]}."
        prompt += f"{pos_information}\n\n"
    elif feature == "dep":
        tuples = []
        for token in doc:
            if token.is_punct:
                continue
            children = [element for element in token.children if not element.is_punct]
            if len(children) == 0:
                continue
            a = f"{token.dep_} ({token.text})"
            b = "  ".join([f"{child.dep_} ({child.text})" for child in children])
            tuples.append(f"{a} -> {b}")
        prompt += "\nTyped dependency structure\n"
        prompt += "We have the following dependencies:\n"
        prompt += "\n".join(tuples) + "\n\n"
        # prompt += "\n".join(tuples[:-1]) + f"\nand\n{tuples[-1]}.\n\n"
    elif feature == "ner":
        ner = [
            f"{ent.text} is {'an' if any([ent.label_.startswith(vowel) for vowel in ['A', 'E', 'I', 'O']]) else 'a'} {ent.label_}"
            for ent in doc.ents
        ]
        if len(ner) == 0:
            pass
        else:
            if len(ner) == 1:
                ner_information = ner[0]
            else:
                ner_information = f"{', '.join(ner[:-1])} and {ner[-1]}"
            prompt += "\nNamed Entities tags\n"
            prompt += f"{ner_information}\n\n"
    elif feature == "none":
        prompt + "\n"

    prompt += f"{tgt} translation\n"
    return prompt
