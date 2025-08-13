REFINE = """
Source: {source}
Translation: {prev_translation}
Please give me a better {tgt} translation without any explanation.
"""

REFINE_V2 = """
We want to translate in {tgt} the following {src} sentence:
{source}
We have a candidate {tgt} translation that we aim to improve:
{prev_translation}

For this purpose we will use the following criteria:
1. The translation should not be empty. Make sure to output words relevant to the information expressed in the sentence.
2. The translation should be written in the correct language, here {tgt}.
3. The translation should be a grammatically correct {tgt} sentence, respecting the grammatical rules and standards of the {tgt} language.
4. The translation should be fluent and convey the same meaning as the original sentence. That means if we translate the translation back into {src}, we should obtain a sentence with the same meaning as the original sentence.

Please give me a better {tgt} translation without any explanation.
"""

def get_refine_prompt(
    source: str, 
    prev_translation: str, 
    src: str,
    tgt: str,
):
    """
    Build a refine-prompt in order to get a model to improve its translation
    Arguments
    ---------
        - source : str,
            Sentence written that we would like to translate
        - prev_translation: str,
            Attempt of translation that we would like to improve
        - src : str,
            Source language
        - tgt : str,
            Target language
    """
    return REFINE.strip().format(source = source, prev_translation = prev_translation, tgt = tgt)
    # return REFINE_V2.strip().format(source = source, prev_translation = prev_translation, src = src, tgt = tgt)