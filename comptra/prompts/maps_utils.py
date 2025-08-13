from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        help="Name or path of the model to use for the generation.",
        default="facebook/nllb-200-distilled-600M",
    )
    return parser.parse_args()


CODES = [
    "amh_Ethi",
    "asm_Beng",
    "bem_Latn",
    "fij_Latn",
    "guj_Gujr",
    "hau_Latn",
    "ibo_Latn",
    "mal_Mlym",
    "som_Latn",
    "tam_Taml",
    "tel_Telu",
    "tir_Ethi",
    "tso_Latn",
    "twi_Latn",
    "tsn_Latn",
    "uig_Arab",
    "wol_Latn",
    "xho_Latn",
    "yor_Latn",
    "mya_Mymr",
    "ceb_Latn",
    "hau_Latn",
    "jav_Latn",
    "kan_Knda",
    "khm_Khmr",
    "kin_Latn",
    "lao_Laoo",
    "mri_Latn",
    "mar_Deva",
    "smo_Latn",
    "sna_Latn",
    "sin_Sinh",
    "sun_Latn",
    "swh_Latn",
    "tuk_Latn",
    "tsn_Latn",
    "urd_Arab",
    "uzn_Latn",
    "plt_Latn",
    "npi_Deva"
]

if __name__ == "__main__":
    args = parse_args()
    model_name_or_path = args.model_name_or_path
    output_path = os.path.join(
        os.path.join(os.path.dirname(__file__), ".."),
        "data/maps"
    )
    # model_name_or_path = "facebook/nllb-moe-54b"
    model_name_or_path = "facebook/nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path, device_map="auto")

    sents = [
        "Around the same time, Patrick Brown, a professor of biochemistry at Stanford University School of Medicine, became interested in developing new techniques for mapping genes.",
        "Libyan security officials say the Afriqiyah Airways plane was flying from Johannesburg, South Africa, Wednesday morning when it crashed short of the runway at the Tripoli airport.",
        "The victory completed a triumphant first season in charge for 38 - year - old Barca coach Pep Guardiola.",
        'On Friday night, protests continued in "an almost celebratory manner" near the QuikTrip until police arrived at around 11:00 p.m.',
        "You may access Bing-powered experiences when using other non-Microsoft services, such as those from Yahoo!",
    ]

    topics = [
        ["Stanford University", "School of Medicine"],
        ["JAS 39C Gripen", "commercial flights"],
        ["Barça", "Sevilla"],
        ["Whitehall", "Downing Street", "Prime Minister's official residence"],
        ["Yahoo!", "Microsoft"],
    ]

    sents_tensor = tokenizer(sents, return_tensors="pt", padding=True).to(model.device)
    topics_tensors = [
        tokenizer(topic, return_tensors="pt", padding=True).to(model.device)
        for topic in topics
    ]
    generation_kwargs = {
        "max_new_tokens": 150,
        "temperature": 1.0,
        "num_beams": 4,
        "top_p": 1.0,
        "repetition_penalty": 1.02,
        "do_sample": False,
    }
    for code in CODES:
        if os.path.exists(os.path.join(output_path, f"{code}.jsonl")):
            continue
        translated_tokens = model.generate(
            **sents_tensor,
            **generation_kwargs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids(code),
        )
        translations = tokenizer.batch_decode(
            translated_tokens, skip_special_tokens=True
        )
        for i, translation in enumerate(translations):
            print(f"{i+1} -> {translation}")

        L = []
        for topics_tensor in topics_tensors:
            tokens = model.generate(
                **topics_tensor,
                **generation_kwargs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(code),
            )
            trans = tokenizer.batch_decode(tokens, skip_special_tokens=True)
            L.append(trans)

        prompt = ""
        for i in range(len(L)):
            prompt += f"{i+1}\nSource: {sents[i]}\nTarget: {translations[i]}\nTopics: {L[i]}\n\n"
        prompt = f"===\n{prompt.strip()}\n==="
        print(prompt)
        with open(os.path.join(output_path, f"{code}.jsonl"), "a") as fout:
            for i in range(len(translations)):
                fout.write(
                    json.dumps({"sentence": translations[i], "keywords": L[i]}) + "\n"
                )
    print("END")
