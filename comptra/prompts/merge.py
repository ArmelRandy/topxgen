from typing import List

WITH_REFINE = """
I would like you to provide a high quality {tgt} translation of a long sentence written in {src}. In order for you to achieve this, you will be provided with a list of words/short sentences and their translation as premises.
These words/short sentences share some words and/or expressions (chunk of words) with the long sentence. Moreover, these words and/or expressions that can be found in common are also used in a similar context in both sentences.
Thus, you can use the translation of the words in the short sentences to infer the translation of the long sentence.

The translations provided as premises are not perfect, I strongly advise you to refine them first in order to make them better correspond to the context of the long sentence before using them to derive a high quality {tgt} translation of the long sentence.

<Output structure>
I. Refining of the premises' translations
<Reason and refine the premises' translations (in order) to make them better correspond to the long sentence's context if it is not already the case.>
II. Recombination and translation
<Think step by step and take inspiration from the premises' refined translations to get a high-quality {tgt} translation of the long sentence. The translation should convey as precisely as possible the information contained in the long sentence written in {src}.>
III. Final translation
The {tgt} translation of the long sentence is thus:
<Final translation>
</Output structure>

If no premises are provided, directly translate the long sentence into {tgt}.
Make sure to carefully follow the above output structure and don't write anything after the final translation!
"""

WITHOUT_REFINE = """
I would like you to provide a high-quality {tgt} translation of a long sentence written in {src}.
In order for you to achieve this, you will be provided with a list of words/short sentences and their translation as premises.
These words/short sentences share some words and/or expressions (chunk of words) with the long sentence. Moreover, these words and/or expressions that can be found in common are also used in a similar context in both sentences.
Thus, you can use the translation of the words in the short sentences to infer the translation of the long sentence.
Combine the premises' translations to derive a high quality translation of the long sentence.

<Output structure>
I. Combination and translation
<Reason and take inspiration from the premises' translations (in order) to get a translation of the long sentence. The translation should convey as precisely as possible the information contained in the long sentence.>
II. Final translation
The {tgt} translation of the long sentence is thus:
<Final translation>
</Output structure>

If no premises are provided, skip the step I and directly provide a high-quality translation of the long sentence.
Make sure to carefully follow this structure and don't write anything after the final translation!
"""

VANILLA = """
Given the following sentence-translation pairs written by a professional translator:

<Demonstrations>
{demonstrations}
</Demonstrations>

Please write a high-quality {tgt} translation of the following {src} sentence

{sentence}

Please make sure to consider the above information and provide only the translation, nothing more.
"""

ZS = """
Please write a high-quality {tgt} translation of the following {src} sentence

{sentence}

Please provide only the translation, nothing more.
"""

def get_merge_prompt(
    sentence: str,
    inputs: List[str],
    outputs: List[str],
    src: str,
    tgt: str,
    template = None,
    method: str = "vanilla",
):
    """
    Build a prompt for the merge of the translations.
    Arguments
    ---------
        - sentence : str,
            Sequence that was decomposed
        - inputs : List[str]
            List of subparts of `x`.
        - outputs : List[str],
            List of translation of the subparts of `x`.
        - src : str,
            Source language (e.g. English)
        - tgt : str,
            Target language (e.g. French)
        - template : Template,
            Template used for MT
        - Method: str,
            Which prompt to use for the merge
    Returns
    -------
        - str :
            Prompt for the Machine Translation of `x` from `src` to `tgt` using its subparts `inputs` tranlated into `outputs`.
    """
    assert len(inputs) == len(
        outputs
    ), f"The number of inputs ({len(inputs)}) should match the number of outputs {(len(outputs))}"
    if template:
        template.header = ""
    demos = ""
    if template:
        for i, (source, target) in enumerate(zip(inputs, outputs)):
            demos += f"{i+1}. {template.prefix}{source}{template.middle}{target}"
            demos += "\n\n"
    else:
        for i, (source, target) in enumerate(zip(inputs, outputs)):
            demos += (
                f"{i+1}. {src} sentence\n{source}\n{tgt} translation\n{target}"
            )
            demos += "\n\n"
    if demos.strip() == "":
        return ZS.strip().format(src=src, tgt=tgt, sentence=sentence)
    
    if method in ["refine", "norefine"]:
        prompt = (
            WITH_REFINE.strip().format(src=src, tgt=tgt)
            if method == "refine"
            else WITHOUT_REFINE.strip().format(src=src, tgt=tgt)
        )
        prompt += "\n\nGiven the following premises,\n"
        prompt += f"Let's infer the {tgt} translation of the following long sentence\n\n{sentence}"
    elif method == "vanilla":
        prompt = VANILLA.strip().format(
            src=src, tgt=tgt, demonstrations=demos.strip(), sentence=sentence
        )
    else:
        raise ValueError(f"Unsupported merge method {method}.")
    return prompt


def extract_translation(sentence: str, method: str):
    """Extract the translation from the output of a merge operation"""
    if method in ["refine", "norefine"]:
        trigger = "sentence is thus:\n"
        if trigger in sentence:
            return (
                sentence[sentence.find(trigger) + len(trigger) :].strip().split("\n")[0]
            )
        trigger = (
            "III. Final translation"
            if method == "norefine"
            else "II. Final translation"
        )
        if trigger in sentence:
            return (
                sentence[sentence.find(trigger) + len(trigger) :].strip().split("\n")[0]
            )
        else:
            print("Trigger not found")
            return sentence.strip().split("\n")[0]
    elif method == "vanilla":
        return sentence.strip().split("\n")[0]
    else:
        pass


if __name__ == "__main__":
    pass
