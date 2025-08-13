TRANSLATE = """
Please provide the {tgt_lan} translation for the {src_lan} sentences:
Example:
Source: {src_example_1} Target: {tgt_example_1}
Source: {src_example_2} Target: {tgt_example_2}
Source: {src_example_3} Target: {tgt_example_3}
Source: {src_example_4} Target: {tgt_example_4}
Source: {src_example_5} Target: {tgt_example_5}
Source: {origin}
Target:
""".strip()

ESTIMATE = """
Please identify errors and assess the quality of the translation.
The categories of errors are accuracy (addition, mistranslation, omission, untranslated text), fluency (character encoding, grammar, inconsistency, punctuation, register, spelling),
locale convention (currency, date, name, telephone, or time format) style (awkward), terminology (inappropriate for context, inconsistent use), non-translation, other, or no-error.
Each error is classified as one of three categories: critical, major, and minor. Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technical errors but do not disrupt the flow or hinder comprehension.

Example1:
Chinese source: 大众点评乌鲁木齐家居商场频道为您提供居然之家地址，电话，营业时间等最新 商户信息， 找装修公司，就上大众点评
English translation: Urumqi Home Furnishing Store Channel provides you with the latest business information such as the address, telephone number, business hours, etc., of high-speed rail, and find a decoration company, and go to the reviews.
MQM annotations:
critical: accuracy/addition - "of high-speed rail"
major: accuracy/mistranslation - "go to the reviews"
minor: style/awkward - "etc.,"

Example2:
English source: I do apologise about this, we must gain permission from the account holder to discuss an order with another person, I apologise if this was done previously, however, I would not be able to discuss this with yourself without the account holders permission.
German translation: Ich entschuldige mich dafür, wir müssen die Erlaubnis einholen, um eine Bestellung mit einer anderen Person zu besprechen. Ich entschuldige mich, falls dies zuvor geschehen wäre, aber ohne die Erlaubnis des Kontoinhabers wäre ich nicht in der Lage, dies mit dir involvement.
MQM annotations:
critical: no-error
major: accuracy/mistranslation - "involvement"
    accuracy/omission - "the account holder"
minor: fluency/grammar - "wäre"
    fluency/register - "dir"

Example3:
English source: Talks have resumed in Vienna to try to revive the nuclear pact, with both sides trying to gauge the prospects of success after the latest exchanges in the stop-start negotiations.
Czech transation: Ve Vídni se ve Vídni obnovily rozhovory o oživení jaderného paktu, pˇricˇemže obeˇ partaje se snaží posoudit vyhlídky na úspeˇch po posledních výmeˇnách v jednáních.
MQM annotations:
critical: no-error
major: accuracy/addition - "ve Vídni"
    accuracy/omission - "the stop-start"
minor: terminology/inappropriate for context - "partake"

{src_lan} source: {origin}
{tgt_lan} translation: {init_trans}
MQM annotations:
""".strip()

REFINE = """
Please provide the {tgt_lan} translation for the {src_lan} sentences.

Example:
Source: {src_example_1} Target: {tgt_example_1}
Source: {src_example_2} Target: {tgt_example_2}
Source: {src_example_3} Target: {tgt_example_3}
Source: {src_example_4} Target: {tgt_example_4}
Source: {src_example_5} Target: {tgt_example_5}

Now, let’s focus on the following {src_lan}-{tgt_lan} translation pair.
Source: {raw_src}
Target: {raw_mt}
I’m not satisfied with this target, because some defects exist: {estimate_fdb}
Critical errors inhibit comprehension of the text. Major errors disrupt the flow, but what the text is trying to say is still understandable. Minor errors are technical errors but do not disrupt the flow or hinder comprehension.

Upon reviewing the translation examples and error information, please proceed to compose the final {tgt_lan} translation to the sentence: {raw_src}

First, based on the defects information locate the error span in the target segment, comprehend its nature, and rectify it. Then, imagine yourself as a native {tgt_lan} speaker, ensuring that the rectified target segment is not only precise but also faithful to the source segment.
""".strip()


def get_tear_prompts(
    description: str,
    src: str,
    tgt: str,
    source: str = None,
    draft: str = None,
    estimate_fdb: str = None,
    demonstrations = None,
):
    """
    Arguments
    ---------
    - description: str,
        Any of ['translate', 'estimate', 'refine']
    - src: str,
        Source language e.g. English
    - tgt: str,
        Target language e.g. English
    - source: str,
        Source sentence.
    - draft: str,
        Tentative translation.
    - estimate_fdb: str,
        Translation error
    - demonstrations: List[(str, str)]
    """
    header = "Given any instruction, make sure to only return the expected answer, nothing before and nothing after.\n\n"
    if description == "translate":
        if demonstrations is None or not demonstrations:
            prompt = f"Please provide the {tgt} translation for the {src} sentences:\nSource: {source}\nTarget:"
        else:
            prompt = TRANSLATE.format(
                src_lan=src,
                tgt_lan=tgt,
                origin=source,
                src_example_1=demonstrations[0][0],
                tgt_example_1=demonstrations[0][1],
                src_example_2=demonstrations[1][0],
                tgt_example_2=demonstrations[1][1],
                src_example_3=demonstrations[2][0],
                tgt_example_3=demonstrations[2][1],
                src_example_4=demonstrations[3][0],
                tgt_example_4=demonstrations[3][1],
                src_example_5=demonstrations[4][0],
                tgt_example_5=demonstrations[4][1],
            )
    elif description == "estimate":
        prompt = ESTIMATE.format(
            src_lan=src, tgt_lan=tgt, origin=source, init_trans=draft
        )
    elif description == "refine":
        prompt = REFINE.format(
            src_lan=src,
            tgt_lan=tgt,
            raw_src=source,
            raw_mt=draft,
            estimate_fdb=estimate_fdb,
            src_example_1=demonstrations[0][0],
            tgt_example_1=demonstrations[0][1],
            src_example_2=demonstrations[1][0],
            tgt_example_2=demonstrations[1][1],
            src_example_3=demonstrations[2][0],
            tgt_example_3=demonstrations[2][1],
            src_example_4=demonstrations[3][0],
            tgt_example_4=demonstrations[3][1],
            src_example_5=demonstrations[4][0],
            tgt_example_5=demonstrations[4][1],
        )
    else:
        raise ValueError(f"The description {description} is not supported by TEaR. Only `translate`, `estimate` and `refine`.")
    return header + prompt
