from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
import argparse
import os
import torch
from comptra.data.dataset import get_datasets

model_name_or_path = "text_sonar_basic_encoder"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")
t2vec_model = TextToEmbeddingModelPipeline(
    encoder=model_name_or_path, tokenizer=model_name_or_path, device=device
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name_or_path",
        type=str,
        default="ntrex",
        help="Name or path to the dataset to embed",
    )
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    return parser.parse_args()


def main(args):
    A = get_datasets(args.dataset_name_or_path, "English")
    output_path = os.path.join(
        os.path.join(os.path.dirname(__file__), args.dataset_name_or_path), "eng",
    )
    os.makedirs(output_path, exist_ok=True)
    name = "SONAR"
    lang = "eng_Latn"
    os.makedirs(os.path.join(output_path, name), exist_ok=True)
    L_devtest = [example["sentence"] for example in A["devtest"]]
    L_dev = [example["sentence"] for example in A["dev"]]

    dev_emb = t2vec_model.predict(
        L_dev, source_lang=lang, batch_size=args.batch_size, progress_bar=True
    )
    devtest_emb = t2vec_model.predict(
        L_devtest, source_lang=lang, batch_size=args.batch_size, progress_bar=True
    )

    if device == "cpu":
        dev_emb = dev_emb.detach().numpy()
        devtest_emb = devtest_emb.detach().numpy()
    else:
        dev_emb = dev_emb.cpu().detach().numpy()
        devtest_emb = devtest_emb.cpu().detach().numpy()

    dev_emb.tofile(os.path.join(output_path, f"{name}/dev.bin"))
    devtest_emb.tofile(os.path.join(output_path, f"{name}/devtest.bin"))
    print("Done!")


import json
import os


def second(args):
    path = os.path.join(
        os.path.join(
            os.path.dirname(__file__), "..", ".."
        ),  # main directory i.e. compositional-translation
        "out/GENERATIONS/FLORES/bm25s/Meta-Llama-3.1-8B-Instruct",
    )
    L = []
    with open(os.path.join(path, "divide_1.jsonl"), "r") as fin:
        for line in fin:
            L.extend(json.loads(line)["propositions"])
    lang = "eng_Latn"
    L_emb = t2vec_model.predict(
        L, source_lang=lang, batch_size=args.batch_size, progress_bar=True
    )
    if device == "cpu":
        L_emb = L_emb.detach().numpy()
    else:
        L_emb = L_emb.cpu().detach().numpy()
    L_emb.tofile(os.path.join(path, f"translate_1.bin"))
    print("Done!")


if __name__ == "__main__":
    args = parse_args()
    # main(args)
    second(args)
