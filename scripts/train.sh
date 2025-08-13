# Fine-tuning

ARGS="\
    --model_name_or_path meta-llama/Meta-Llama-3-8B\
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B\
    --dataset_name_or_path topxgen\
    --input_column_name source\
    --output_column_name target\
    --max_length 512\
    --max_steps 5000\
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1\
    --gradient_accumulation_steps 16\
    --learning_rate 1e-5\
    --lr_scheduler_type cosine\
    --num_warmup_steps 500\
    --weight_decay 0.01\
    --bf16\
    --seed 122\
    --output_dir /path/to/REVERSE/GEMMA/T=1.0/checkpoints-llama-3-8b-full-eus-wiki\
    --logging_steps 100\
    --eval_steps 200\
    --save_steps 200\
    --src English\
    --target_languages Basque\
    --dataset_size -1\
    --strategy soonest\
    --targets_only\
    --data_dir data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B\
    --reverse\
    --gradient_checkpointing\
    --test_size_ratio 1000\
    "
# With LoRA
ARGS="\
    --model_name_or_path meta-llama/Meta-Llama-3-8B\
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B\
    --dataset_name_or_path topxgen\
    --input_column_name source\
    --output_column_name target\
    --max_length 512\
    --max_steps 5000\
    --per_device_train_batch_size 1\
    --per_device_eval_batch_size 1\
    --gradient_accumulation_steps 16\
    --learning_rate 1e-5\
    --lr_scheduler_type cosine\
    --num_warmup_steps 500\
    --weight_decay 0.01\
    --bf16\
    --seed 122\
    --output_dir /path/to/REVERSE/GEMMA/T=1.0/checkpoints-llama-3-8b-full-eus-wiki\
    --logging_steps 100\
    --eval_steps 200\
    --save_steps 200\
    --src English\
    --target_languages Basque\
    --dataset_size -1\
    --strategy soonest\
    --targets_only\
    --data_dir data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B\
    --reverse\
    --gradient_checkpointing\
    --test_size_ratio 1000\
    --lora_r 32\
    --lora_alpha 64\
    --lora_dropout 0.05\
    --target_modules q_proj k_proj v_proj o_proj\
    --use_peft\
    "

echo "SLURM_ARRAY_TASK_ID is: $SLURM_ARRAY_TASK_ID"
list_of_suffixes=("topxgen" "knn-instruct" "self-instruct")
list_of_codes=(XGEN KNN SI)
MASTER_PORT=$((27500 + 100*$SLURM_ARRAY_TASK_ID))
SUFFIX=${list_of_suffixes[($SLURM_ARRAY_TASK_ID - 1)]}
CODE=${list_of_codes[($SLURM_ARRAY_TASK_ID - 1)]}

ARGS2="\
    --model_name_or_path meta-llama/Meta-Llama-3-8B\
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B\
    --dataset_name_or_path topxgen\
    --input_column_name source\
    --output_column_name target\
    --max_length 512\
    --max_steps 5000\
    --per_device_train_batch_size 4\
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 4\
    --learning_rate 1e-5\
    --lr_scheduler_type cosine\
    --num_warmup_steps 500\
    --weight_decay 0.01\
    --bf16\
    --seed 122\
    --output_dir /path/to/REVERSE/GEMMA/T=1.0/${CODE}/checkpoints-${T2}-full-som\
    --logging_steps 100\
    --eval_steps 200\
    --save_steps 200\
    --src English\
    --target_languages Somali\
    --dataset_size 21000\
    --strategy soonest\
    --targets_only\
    --data_dir data/wiki/gemma-3-27b-it/${SUFFIX}/T=1.0/nllb-200-3.3B\
    --reverse\
    --gradient_checkpointing\
    --test_size_ratio 1000\
    "

echo "SLURM_ARRAY_TASK_ID is: $SLURM_ARRAY_TASK_ID"
list_of_names=("flores" "smol" "topxgen")
list_of_codes=(FLORES SMOL XGEN)
MASTER_PORT=$((28500 + 100*$SLURM_ARRAY_TASK_ID))
NAME=${list_of_names[($SLURM_ARRAY_TASK_ID - 1)]}
CODE=${list_of_codes[($SLURM_ARRAY_TASK_ID - 1)]}

ARGS3="\
    --model_name_or_path meta-llama/Meta-Llama-3-8B\
    --tokenizer_name_or_path meta-llama/Meta-Llama-3-8B\
    --dataset_name_or_path ${NAME}\
    --input_column_name source\
    --output_column_name target\
    --max_length 512\
    --max_steps 1000\
    --per_device_train_batch_size 2\
    --per_device_eval_batch_size 2\
    --gradient_accumulation_steps 1\
    --learning_rate 1e-5\
    --lr_scheduler_type cosine\
    --num_warmup_steps 100\
    --weight_decay 0.01\
    --bf16\
    --seed 122\
    --output_dir path/to/REVERSE/GEMMA/REAL/${CODE}/checkpoints-${T2}-full-ibo\
    --logging_steps 50\
    --eval_steps 100\
    --save_steps 100\
    --src English\
    --target_languages Igbo\
    --dataset_size 863\
    --strategy soonest\
    --targets_only\
    --data_dir data/wiki/gemma-3-27b-it/topxgen/T=1.0/nllb-200-3.3B\
    --reverse\
    --gradient_checkpointing\
    --test_size_ratio 63\
    "