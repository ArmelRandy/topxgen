import argparse
import os

import torch
import random
import warnings
from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import IterableDataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    logging,
    set_seed,
)

"""
Fine-tune a model on an instruction dataset
"""
import spacy

nlp = spacy.load("en_core_web_sm")

PATH = os.path.dirname(__file__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name_or_path", type=str, default="bigcode/starcoderbase-1b"
    )
    parser.add_argument(
        "--tokenizer_name_or_path", type=str, help="Name or path of the tokenizer."
    )
    parser.add_argument(
        "--dataset_name_or_path", type=str, default="HuggingFaceH4/CodeAlpaca_20K"
    )
    parser.add_argument(
        "--subset", type=str, help="subset parameter of `load_dataset`."
    )
    parser.add_argument("--split", type=str, help="split parameter of `load_dataset`.")
    parser.add_argument("--size_valid_set", type=int, help="size of the training set.")
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--shuffle_buffer", type=int, default=5000)
    parser.add_argument("--num_of_sequences", type=int, default=1000)
    parser.add_argument(
        "--test_size_ratio",
        type=float,
        default=0.1,
        help="Proportion of the test set in the dataset.",
    )
    parser.add_argument("--input_column_name", type=str, default="prompt")
    parser.add_argument("--output_column_name", type=str)
    parser.add_argument(
        "--targets_only",
        action="store_true",
        help="Train on answer only during instruction fine-tuning.",
    )  # default value of False

    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument(
        "--max_steps", type=int, default=1000, help="Number of training steps"
    )
    parser.add_argument("--num_train_epochs", type=int, help="Number of epochs")
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        help="The train batch size per device.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size", type=int, help="The eval batch size per device."
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        help="The number of gradient accumulation steps.",
    )
    parser.add_argument(
        "--eos_token_id", type=int, help="eos_token_id if not defined in the tokenizer."
    )
    parser.add_argument("--use_peft", action="store_true", help="Whether to use peft.")
    parser.add_argument(
        "--lora_r", type=int, default=16, help="The lora rank parameter."
    )
    parser.add_argument(
        "--lora_alpha", type=int, default=32, help="The lora alpha parameter."
    )
    parser.add_argument(
        "--lora_dropout", type=float, default=0.05, help="The lora dropout parameter."
    )
    parser.add_argument(
        "--target_modules",
        nargs="+",
        # type=str,
        # default="c_proj c_attn q_attn",
        help="The lora target modules parameter.",
    )

    parser.add_argument(
        "--learning_rate", type=float, default=5e-6, help="Learning rate."
    )
    parser.add_argument(
        "--min_learning_rate", type=float, help="Minimum learning rate if relevant."
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        help="Learning rate scheduler.",
    )
    parser.add_argument(
        "--num_warmup_steps", type=int, default=100, help="Number of warmup steps."
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="Weight decay parameter."
    )

    # parser.add_argument("--local_rank", type=int, default=0)
    # parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--fp16", action="store_true", help="Train in fp16.")
    parser.add_argument("--bf16", action="store_true", help="Train in bf16.")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Perform gradient checkpointing.",
    )
    parser.add_argument("--seed", type=int, default=122, help="seed.")
    parser.add_argument("--num_workers", type=int, default=24)
    parser.add_argument(
        "--output_dir", type=str, help="Where to store the checkpoints."
    )
    parser.add_argument(
        "--logging_steps", default=10, type=int, help="The logging frequency."
    )
    parser.add_argument(
        "--eval_steps", default=10, type=int, help="The saving frequency."
    )
    parser.add_argument(
        "--save_steps", default=50, type=int, help="The evaluation frequency."
    )
    parser.add_argument(
        "--deepspeed", type=str, help="Path to the deepspeed config file."
    )
    parser.add_argument(
        "--use_flash_attn", action="store_true", help="Use flash attention."
    )
    parser.add_argument(
        "--neftune_noise_alpha",
        type=int,
        help="Alpha parameter for neftune fine-tuning.",
    )
    parser.add_argument("--src", type=str, help="Source language")
    parser.add_argument(
        "--target_languages", nargs="+", type=str, help="Target languages"
    )
    parser.add_argument("--push_to_hub", type=str, help="Push to hub")
    parser.add_argument(
        "--percentile",
        type=float,
        help="Which percentile of the dataset to keep for the training.",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        help="Total (train + test) size of the dataset in terms of number of pairs.",
    )
    parser.add_argument(
        "--strategy", type=str, help="Which strategy to choose for sample selection."
    )
    parser.add_argument("--data_dir", type=str, help="Path to the data.")
    parser.add_argument(
        "--load_in_8bit", action="store_true", help="Whether to train in 8bit."
    )
    parser.add_argument(
        "--load_in_4bit", action="store_true", help="Whether to train in 4bit."
    )
    parser.add_argument("--reverse", action="store_true", help="both directions.")
    return parser.parse_args()


def chars_token_ratio(
    dataset, tokenizer, input_column_name, output_column_name, nb_examples=400
):
    """
    Estimate the average number of characters per token in the dataset.
    """
    total_characters, total_tokens = 0, 0
    for _, example in tqdm(zip(range(nb_examples), iter(dataset)), total=nb_examples):
        text = prepare_sample_text(example, input_column_name, output_column_name)
        total_characters += len(text)
        if tokenizer.is_fast:
            total_tokens += len(tokenizer(text).tokens())
        else:
            total_tokens += len(tokenizer.tokenize(text))

    return total_characters / total_tokens


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def prepare_sample_text(
    example, input_column_name, output_column_name, src="English", tgt="French"
):
    """Prepare the text from a sample of the dataset."""
    if output_column_name:
        text = f"Translate this from {src} to {tgt}:\n{src}: {example[input_column_name]}\n{tgt}: {example[output_column_name]}"
    else:
        text = example[input_column_name]
    return text


import numpy as np


def numpy_find(sublist, main_list):
    """
    Finds the starting index of the first occurrence of sublist in main_list using NumPy.

    Args:
        sublist (list): The list to search for.
        main_list (list): The list in which to search.

    Returns:
        int: The starting index of the first occurrence of sublist, or -1 if not found.
    """
    sublist = np.array(sublist)
    main_list = np.array(main_list)

    if len(sublist) == 0:
        return 0  # Empty sublist is found at index 0

    # Create a sliding window view of main_list
    for i in range(len(main_list) - len(sublist) + 1):
        if np.array_equal(main_list[i : i + len(sublist)], sublist):
            return i

    return -1


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        input_column_name,
        output_column_name,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        shuffle=True,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = (
            tokenizer.eos_token_id
            if tokenizer.eos_token_id is not None
            else args.eos_token_id
        )
        self.dataset = dataset
        self.seq_length = seq_length
        self.infinite = infinite
        self.current_size = 0
        self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
        self.input_column_name = input_column_name
        self.output_column_name = output_column_name
        self.shuffle = shuffle

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(
                        prepare_sample_text(
                            next(iterator),
                            self.input_column_name,
                            self.output_column_name,
                        )
                    )
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    examples.append(input_ids)
            if self.shuffle:
                random.shuffle(examples)
            for example in examples:
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(example),
                    "labels": torch.LongTensor(example),
                }


class TLConstantLengthDataset(ConstantLengthDataset):
    """
    Target Loss ConstantLengthDataset
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer_list, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    example = next(iterator)
                    q_str = example[self.input_column_name]
                    a_str = example[self.output_column_name]
                    left = f"Translate this from {example['source_language']} to {example['target_language']}:\n{example['source_language']}: "
                    middle = f"\n{example['target_language']}: "
                    buffer_list.append((left, q_str, middle, a_str))
                    buffer_len += len(left + q_str + middle + a_str)
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                    else:
                        more_examples = False
                        break
            all_token_ids = []
            all_label_ids = []
            for left, q_str, middle, a_str in buffer_list:
                tokenized_input = self.tokenizer(left + q_str + middle + a_str)[
                    "input_ids"
                ]
                question_token_ids = self.tokenizer(left + q_str + middle)["input_ids"]
                assert (
                    tokenized_input[: len(question_token_ids) - 1]
                    == question_token_ids[:-1]
                ), "There is an issue"
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
                all_label_ids.extend(
                    [-100] * (len(question_token_ids) - 1)
                    + tokenized_input[len(question_token_ids) - 1 :]
                    + [self.concat_token_id]
                )
            # sanity check
            assert len(all_token_ids) == len(all_label_ids)

            input_examples = []
            output_examples = []
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                label_ids = all_label_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    input_examples.append(input_ids)
                    output_examples.append(label_ids)

            if self.shuffle:
                examples = list(zip(input_examples, output_examples))
                random.shuffle(examples)
                input_examples, output_examples = zip(*examples)
                input_examples, output_examples = (
                    list(input_examples),
                    list(output_examples),
                )

            for input_ids, label_ids in zip(input_examples, output_examples):
                self.current_size += 1
                yield {
                    "input_ids": torch.LongTensor(input_ids),
                    "labels": torch.LongTensor(label_ids),
                }


def create_datasets(tokenizer, args):
    if args.test_size_ratio <= 1:
        test_size_ratio = args.test_size_ratio
    else:
        test_size_ratio = int(args.test_size_ratio)
    print(f"test size ratio: {test_size_ratio}")
    if args.dataset_name_or_path == "flores":
        from train_datasets import get_flores

        dataset = get_flores(
            src=args.src,
            languages=args.target_languages,
            test_size_ratio=test_size_ratio,
            seed=args.seed,
            size=args.dataset_size,
        )
        train_data, valid_data = dataset["train"], dataset["test"]
        print("FLORES")
        print(dataset)
    elif args.dataset_name_or_path == "smol":
        from train_datasets import get_smol

        dataset = get_smol(
            src=args.src,
            languages=args.target_languages,
            test_size_ratio=test_size_ratio,
            seed=args.seed,
        )
        train_data, valid_data = dataset["train"], dataset["test"]
        print("SMOL")
        print(dataset)
    elif args.dataset_name_or_path == "ours":
        from train_datasets import get

        dataset = get(
            data_dir=args.data_dir if args.data_dir else os.path.join(PATH, "data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B"),
            languages=args.target_languages,
            percentile=args.percentile,
            size=args.dataset_size,
            strategy=args.strategy,
            test_size_ratio=test_size_ratio,
            seed=args.seed,
            reverse=args.reverse,
        )
        print("OURS")
        print(dataset, dataset["train"][0])
        train_data, valid_data = dataset["train"], dataset["test"]
    else:
        dataset = load_dataset(
            args.dataset_name_or_path,
            data_dir=args.subset,
            split=args.split,
            num_proc=args.num_workers if not args.streaming else None,
            streaming=args.streaming,
        )
        if args.streaming:
            print("Loading the dataset in streaming mode")
            valid_data = dataset.take(args.size_valid_set)
            train_data = dataset.skip(args.size_valid_set)
            train_data = train_data.shuffle(
                buffer_size=args.shuffle_buffer, seed=args.seed
            )
        else:
            try:
                train_data = dataset["train"]
                valid_data = dataset["test"]
            except:
                dataset = dataset.train_test_split(test_size=args.size_valid_set)
                train_data = dataset["train"]
                valid_data = dataset["test"]
        return train_data, valid_data

    if args.dataset_name_or_path in ["smol", "flores"] and args.reverse:
        print("REVERSING...")
        from datasets import Dataset, concatenate_datasets

        reverse_ds_train = Dataset.from_dict(
            {
                "source": train_data["target"],
                "target": train_data["source"],
                "source_language": train_data["target_language"],
                "target_language": train_data["source_language"],
            }
        )
        reverse_ds_test = Dataset.from_dict(
            {
                "source": valid_data["target"],
                "target": valid_data["source"],
                "source_language": valid_data["target_language"],
                "target_language": valid_data["source_language"],
            }
        )
        train_data = concatenate_datasets([train_data, reverse_ds_train]).shuffle(
            seed=args.seed
        )
        valid_data = concatenate_datasets([valid_data, reverse_ds_test]).shuffle(
            seed=args.seed
        )
        print(train_data, valid_data)
    if not args.output_column_name:
        warnings.warn(
            "You did not provide a output column name. If you're not going to work on 2 columns, ignore this warning."
        )

    chars_per_token = chars_token_ratio(
        train_data,
        tokenizer,
        args.input_column_name,
        args.output_column_name,
    )
    print(f"The character to token ratio of the dataset is: {chars_per_token:.2f}")

    if args.targets_only:
        train_dataset = TLConstantLengthDataset(
            tokenizer,
            train_data,
            infinite=True,
            seq_length=args.max_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )
        valid_dataset = TLConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=args.max_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )

    else:
        train_dataset = ConstantLengthDataset(
            tokenizer,
            train_data,
            infinite=True,
            seq_length=args.max_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )
        valid_dataset = ConstantLengthDataset(
            tokenizer,
            valid_data,
            infinite=False,
            seq_length=args.max_length,
            chars_per_token=chars_per_token,
            input_column_name=args.input_column_name,
            output_column_name=args.output_column_name,
            num_of_sequences=args.num_of_sequences,
        )
    return train_dataset, valid_dataset


def run_training(args, train_data, val_data):
    if args.deepspeed:
        print(f"Does {args.deepspeed} exist? {os.path.exists(args.deepspeed)}")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        # evaluation_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=(
            args.num_train_epochs if args.num_train_epochs is not None else -1
        ),
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=(
            {"min_lr": args.min_learning_rate}
            if args.min_learning_rate is not None
            else {}
        ),
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": True} if args.gradient_checkpointing else None
        ),
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        run_name=f"{args.model_name_or_path.split('/')[-1]}-{args.src}-to-{args.dataset_name_or_path}",
        report_to="wandb",
        ddp_find_unused_parameters=False,
        deepspeed=args.deepspeed,
        push_to_hub=args.push_to_hub,
    )

    print(f"Loading the model: {args.model_name_or_path}")

    if args.use_peft:
        print(f"We use PEFT!")
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            use_auth_token=True,
            # use_cache=not args.gradient_checkpointing,
            torch_dtype=torch.bfloat16,
            device_map=(
                {"": Accelerator().process_index}
                if any([args.load_in_4bit, args.load_in_8bit])
                else None
            ),
            # device_map="auto",
            # device_map="cuda",
            trust_remote_code=True,
            use_flash_attention_2=args.use_flash_attn,
            # use_flash_attn=args.use_flash_attn,
            load_in_8bit=args.load_in_8bit and not args.load_in_4bit,
            load_in_4bit=args.load_in_4bit,
        )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=(
                args.target_modules.split(" ")
                if isinstance(args.target_modules, str)
                else args.target_modules
            ),
        )

        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)

        if any([args.load_in_4bit, args.load_in_8bit]):
            from peft import prepare_model_for_kbit_training

            model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            use_auth_token=True,
            # use_cache=not args.gradient_checkpointing,
            torch_dtype=torch.bfloat16,
            # device_map={"": Accelerator().process_index},
            # device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=args.use_flash_attn,
            # use_flash_attn=args.use_flash_attn,
            load_in_8bit=args.load_in_8bit and not args.load_in_4bit,
            load_in_4bit=args.load_in_4bit,
        )

    # print_trainable_parameters(model)

    train_data.start_iteration = 0

    print("Starting main loop")

    if args.neftune_noise_alpha is None:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
        )
    else:
        from trl import SFTTrainer

        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=val_data,
            packing=True,
            neftune_noise_alpha=args.neftune_noise_alpha,
        )
        
    from transformers import GenerationConfig

    trainer.model.generation_config = GenerationConfig(temperature=None, top_p=None)

    print("Training...")
    trainer.train(
        resume_from_checkpoint=(
            os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0
        )
    )

def seq_to_seq(args):
    from transformers import (
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        AutoModelForSeq2SeqLM,
    )
    from comptra.languages import MAPPING_LANG_TO_KEY

    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name_or_path, device_map="auto"
    )
    tokenizer_name_or_path = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path
        else args.model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path,
        use_auth_token=True,
        trust_remote_code=True,
        padding_side="left",
    )
    print(f"Model: {args.model_name_or_path}")
    from train_datasets import get

    if args.test_size_ratio <= 1:
        test_size_ratio = args.test_size_ratio
    else:
        test_size_ratio = int(args.test_size_ratio)
    print(f"test size ratio: {test_size_ratio}")
    if args.dataset_name_or_path == "flores":
        from train_datasets import get_flores

        dataset = get_flores(
            src=args.src,
            languages=args.target_languages,
            test_size_ratio=test_size_ratio,
            seed=args.seed,
            size=args.dataset_size,
        )
        train_dataset, eval_dataset = dataset["train"], dataset["test"]
        print("FLORES")
        print(dataset)
    elif args.dataset_name_or_path == "smol":
        from train_datasets import get_smol

        dataset = get_smol(
            src=args.src,
            languages=args.target_languages,
            test_size_ratio=test_size_ratio,
            seed=args.seed,
        )
        train_dataset, eval_dataset = dataset["train"], dataset["test"]
        print("SMOL")
        print(dataset)
        
    elif args.dataset_name_or_path == "ours":
        dataset = get(
            data_dir=args.data_dir if args.data_dir else os.path.join(PATH, ""),
            languages=args.target_languages,
            percentile=args.percentile,
            size=args.dataset_size,
            strategy=args.strategy,
            test_size_ratio=test_size_ratio,
            seed=args.seed,
            reverse=args.reverse,
        )
        print("OURS")
        print(dataset, dataset["train"][0])
        train_dataset, eval_dataset = dataset["train"], dataset["test"]

    def apply_translation_template(example):
        q_str = example[args.input_column_name]
        a_str = example[args.output_column_name]
        tokenizer.src_lang = MAPPING_LANG_TO_KEY[example["source_language"]]
        tokenizer.tgt_lang = MAPPING_LANG_TO_KEY[example["target_language"]]
        return tokenizer(
            q_str,
            text_target=a_str,
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )

    train_dataset = train_dataset.map(apply_translation_template)
    eval_dataset = eval_dataset.map(apply_translation_template)

    train_dataset = train_dataset.remove_columns(
        [
            args.input_column_name,
            args.output_column_name,
            "source_language",
            "target_language",
        ]
    )
    eval_dataset = eval_dataset.remove_columns(
        [
            args.input_column_name,
            args.output_column_name,
            "source_language",
            "target_language",
        ]
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        dataloader_drop_last=True,
        # evaluation_strategy="steps",
        eval_strategy="steps",
        save_strategy="steps",
        num_train_epochs=(
            args.num_train_epochs if args.num_train_epochs is not None else -1
        ),
        max_steps=args.max_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type=args.lr_scheduler_type,
        lr_scheduler_kwargs=(
            {"min_lr": args.min_learning_rate}
            if args.min_learning_rate is not None
            else {}
        ),
        warmup_steps=args.num_warmup_steps,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs=(
            {"use_reentrant": True} if args.gradient_checkpointing else None
        ),
        fp16=args.fp16,
        bf16=args.bf16,
        weight_decay=args.weight_decay,
        # run_name=f"{args.model_name_or_path.split('/')[-1]}-{args.src}-to-{args.tgt}",
        run_name=f"{args.model_name_or_path.split('/')[-1]}-{args.src}-to-{args.dataset_name_or_path}",
        report_to="wandb",
        ddp_find_unused_parameters=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train(
        resume_from_checkpoint=(
            os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 0
        )
    )


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_name_or_path = (
        args.tokenizer_name_or_path
        if args.tokenizer_name_or_path
        else args.model_name_or_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path, use_auth_token=True, trust_remote_code=True
    )
    print(f"Model: {args.model_name_or_path}")
    train_dataset, eval_dataset = create_datasets(tokenizer, args)
    run_training(args, train_dataset, eval_dataset)


if __name__ == "__main__":
    args = get_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    logging.set_verbosity_error()
    
    if "nllb" in args.model_name_or_path:
        seq_to_seq(args)
        exit()

    main(args)
