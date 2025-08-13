from train_datasets import get, get_smol, get_flores
from transformers import AutoTokenizer
from vendi_score import vendi
from sklearn.metrics import pairwise_distances
import numpy as np
from tqdm import tqdm
# Diversity vs SI vs XGEN into English
from datasets import concatenate_datasets, Dataset

def get_vendi_score():
    paths = [
        os.path.join(os.path.dirname(__filename), f"data/wiki/gemma-3-27b-it/self-instruct/T=1.0/nllb-200-3.3B"),
        os.path.join(os.path.dirname(__filename), f"data/wiki/gemma-3-27b-it/knn-instruct/T=1.0/nllb-200-3.3B"),
        os.path.join(os.path.dirname(__filename), f"data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B")
    ]

    names = [
        "SI",
        "KNN",
        "XGEN"
    ]

    from sonar.inference_pipelines.text import TextToEmbeddingModelPipeline
    from sonar.models.blaser.loader import load_blaser_model

    text_embedder = TextToEmbeddingModelPipeline(
        encoder="text_sonar_basic_encoder", tokenizer="text_sonar_basic_encoder"
    )

    MP = {
        "Basque": "eus_Latn",
        "Hausa": "hau_Latn",
        "Igbo": "ibo_Latn",
        "Kinyarwanda": "kin_Latn",
        "Nepali": "npi_Deva",
        "Somali": "som_Latn",
        "Sundanese": "sun_Latn",
        "Swahili": "swh_Latn",
        "Urdu": "urd_Arab",
        "Xhosa": "xho_Latn",
        "English": "eng_Latn"
    }

    batch_size = 2048
    K = 20000

    for language in list(MP.keys())[:-1]:
        for p, path in enumerate(paths):
            try:    
                ds = get(data_dir=path, languages=[language], percentile=None, size=K, strategy="soonest", test_size_ratio=0.0, seed=122, reverse=False)
            except Exception as e:
                print(f"Error loading {names[p]} dataset for {language}: {e}")
                continue
            ds = get(data_dir=path, languages=[language], percentile=None, size=K, strategy="soonest", test_size_ratio=0.0, seed=122, reverse=False)
            print(ds, len(ds))
            print(f"{'-'*20}\n{names[p]}\n{'-'*20}\n{language}\n{'-'*20}")
            L_src = []
            L_tgt = []
            for i in tqdm(range(0, len(ds), batch_size)):
                batch_sentences = ds[i : i + batch_size]
                # print(batch_sentences)
                src_embs = text_embedder.predict(
                    #[example["source"] for example in batch_sentences], source_lang="eng_Latn"
                    batch_sentences["source"], source_lang="eng_Latn", batch_size=64, progress_bar=True
                )
                tgt_embs = text_embedder.predict(
                    #[example["target"] for example in batch_sentences], source_lang=MP[language]
                    batch_sentences["target"], source_lang=MP[language], batch_size=64, progress_bar=True
                )
                L_src.append(src_embs)
                L_tgt.append(tgt_embs)
            L_src = np.concatenate(L_src, axis=0)
            L_tgt = np.concatenate(L_tgt, axis=0)
            L_src = L_src.reshape(-1, 1024)
            L_tgt = L_tgt.reshape(-1, 1024)
            K_src = pairwise_distances(L_src, L_src, metric="cosine")
            print(f"vendi score src (English): {names[p]}: {vendi.score_K(K_src)}")
            K_tgt = pairwise_distances(L_tgt, L_tgt, metric="cosine")
            print(f"vendi score tgt ({language}): {names[p]}: {vendi.score_K(K_tgt)}")

    for language in list(MP.keys())[:-1]:
        print(f"{'-'*20}\n{language}\n{'-'*20}")
        for name in ["SMOL", "FLORES"]:
            try:
                if name == "SMOL":
                    dataset = get_smol(src="English", languages=[language], test_size_ratio=63, seed=122)
                else:
                    dataset = get_flores(src="English", languages=[language], test_size_ratio=63, seed=122, size=-1)
            except Exception as e:
                print(f"Error loading {name} dataset for {language}: {e}")
                continue
            ds = concatenate_datasets([dataset["train"], dataset["test"]])
            L_src = []
            L_tgt = []
            print(ds)
            for i in tqdm(range(0, len(ds), batch_size)):
                batch_sentences = ds[i : i + batch_size]
                src_embs = text_embedder.predict(
                    #[example["source"] for example in batch_sentences], source_lang="eng_Latn"
                    batch_sentences["source"], source_lang="eng_Latn", batch_size=64, progress_bar=True
                )
                tgt_embs = text_embedder.predict(
                    #[example["target"] for example in batch_sentences], source_lang=MP[language]
                    batch_sentences["target"], source_lang=MP[language], batch_size=64, progress_bar=True
                )
                L_src.append(src_embs)
                L_tgt.append(tgt_embs)
            L_src = np.concatenate(L_src, axis=0)
            L_tgt = np.concatenate(L_tgt, axis=0)
            L_src = L_src.reshape(-1, 1024)
            L_tgt = L_tgt.reshape(-1, 1024)
            K_src = pairwise_distances(L_src, L_src, metric="cosine")
            print(f"vendi score src: {name}: {vendi.score_K(K_src)}")
            K_tgt = pairwise_distances(L_tgt, L_tgt, metric="cosine")
            print(f"vendi score tgt: {name}: {vendi.score_K(K_tgt)}")


from comptra.evaluate.metricx24.models import MT5ForRegression
from transformers import TrainingArguments, Trainer
import torch

def get_qe_scores():
    batch_size = 1
    # Check if GPU is available and set the device accordingly
    if torch.cuda.is_available():
        device = torch.device("cuda")
        per_device_batch_size = batch_size // torch.cuda.device_count()
    else:
        device = torch.device("cpu")
        per_device_batch_size = batch_size

    print(f"Using device: {device}")

    model = MT5ForRegression.from_pretrained("google/metricx-24-hybrid-xxl-v2p6", torch_dtype=torch.bfloat16)
    model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("google/mt5-xl")

    training_args = TrainingArguments(
        output_dir="out",
        per_device_eval_batch_size=per_device_batch_size,
        dataloader_pin_memory=False,
        report_to = "none"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
    )

    # Datasets utilities
    def _make_input(example, is_qe=True):
        if is_qe:
            example["input"] = (
                "candidate: "
                + example["hypothesis"]
                + " source: "
                + example["source"]
            )
        else:
            example["input"] = (
                "candidate: "
                + example["hypothesis"]
                + " reference: "
                + example["reference"]
            )
        return example
        
    max_input_length = 1024
    def _tokenize(example):
        return tokenizer(
            example["input"],
            max_length=max_input_length,
            truncation=True,
            padding=False,
        )

    def _remove_eos(example):
        example["input_ids"] = example["input_ids"][:-1]
        example["attention_mask"] = example["attention_mask"][:-1]
        return example

    model_name = "meta-llama/Meta-Llama-3-8B"
    """
    model_name = "meta-llama/Llama-2-7b-hf"
    model_name = "google/gemma-3-27b-it"
    model_name = "facebook/nllb-200-3.3B"
    """
    print(f"Loading tokenizer from {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    path = os.path.join(os.path.dirname(__filename), f"data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B")

    M = 20000

    for language in [
        "Basque",
        "Hausa",
        "Igbo",
        "Kinyarwanda",
        "Nepali",
        "Somali",
        "Sundanese",
        "Swahili",
        "Urdu",
        "Xhosa",
    ]:
        dataset = get(data_dir=path, languages=[language], percentile=None, size=M, strategy="soonest", test_size_ratio=0.0, seed=122, reverse=False)
        print(dataset, len(dataset))
        # Compute the average length of the sentences in the dataset, source and target
        src_lengths = []
        tgt_lengths = []
        for example in dataset:
            src_lengths.append(len(example["source"].split()))
            tgt_lengths.append(len(example["target"].split()))
        avg_src_length = sum(src_lengths) / len(src_lengths)
        avg_tgt_length = sum(tgt_lengths) / len(tgt_lengths)
        print(f"Average source length for {language}: {avg_src_length}")
        print(f"Average target length for {language}: {avg_tgt_length}")
        # Compute the average number of tokens per sentence in the dataset, source and target
        src_token_lengths = []
        tgt_token_lengths = []
        for example in dataset:
            src_token_lengths.append(len(tok(example["source"])["input_ids"]))
            tgt_token_lengths.append(len(tok(example["target"])["input_ids"]))
        avg_src_token_length = sum(src_token_lengths) / len(src_token_lengths)
        avg_tgt_token_length = sum(tgt_token_lengths) / len(tgt_token_lengths)
        print(f"Average source token length for {language}: {avg_src_token_length}")
        print(f"Average target token length for {language}: {avg_tgt_token_length}")
        # Compute MetricX QE score for the dataset
        print(f"source: {dataset[0]['source']}\ntarget: {dataset[0]['target']}")
        ds = Dataset.from_dict(
            {
                "source": [example["source"] for example in dataset],
                "hypothesis": [example["target"] for example in dataset],
            }
        )
        ds = ds.map(_make_input)
        ds = ds.map(_tokenize)
        ds = ds.map(_remove_eos)
        ds.set_format(
            type="torch",
            columns=["input_ids", "attention_mask"],
            device=device,
            output_all_columns=True,
        )
        print(f"ds: {ds}")
        score_predictions, _, _ = trainer.predict(test_dataset=ds)
        print(f"QE {language}: {np.mean([float(pred) for pred in score_predictions])}")
 
        for name in ["SMOL", "FLORES"]:
            try:
                if name == "SMOL":
                    dataset = get_smol(src="English", languages=[language], test_size_ratio=63, seed=122)
                else:
                    dataset = get_flores(src="English", languages=[language], test_size_ratio=63, seed=122, size=-1)
            except Exception as e:
                print(f"Error loading {name} dataset for {language}: {e}")
                continue
            dataset = concatenate_datasets([dataset["train"], dataset["test"]])
            print(f"source: {dataset[0]['source']}\ntarget: {dataset[0]['target']}")
            ds = Dataset.from_dict(
                {
                    "source": [example["source"] for example in dataset],
                    "hypothesis": [example["target"] for example in dataset],
                }
            )
            ds = ds.map(_make_input)
            ds = ds.map(_tokenize)
            ds = ds.map(_remove_eos)
            ds.set_format(
                type="torch",
                columns=["input_ids", "attention_mask"],
                device=device,
                output_all_columns=True,
            )
            score_predictions, _, _ = trainer.predict(test_dataset=ds)
            print(f"{name} {language}: {np.mean([float(pred) for pred in score_predictions])}")

if __name__ == "__main__":
    pass