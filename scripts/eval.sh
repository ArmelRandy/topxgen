ARGS="
    --model_name_or_path google/metricx-24-hybrid-xxl-v2p6\
    --dataset_name_or_path flores\
    --max_input_length 1024\
    --batch_size 1\
    --data_dir /path/to/GENERATIONS/FLORES\
    --number_of_predictions 1012\
    --seed 122\
    --num_workers 8\
    --metric metricx\
    --languages English Basque Hausa Igbo Kinyarwanda Nepali Somali Sundanese Swahili Urdu Xhosa\
    --strategies BENCHMARKING\
    --names Llama-3.3-70B-Instruct llama-2-7b-bs-mono llama-2-7b-greedy-mono llama-3-8b-bs-mono llama-3-8b-greedy-mono llama-3-8b-greedy-multi llama-3-8b-bs-multi LLaMAX3-8B-Alpaca LLaMAX2-7B-Alpaca nllb-200-3.3B Meta-Llama-3.1-70B-Instruct Meta-Llama-3-8B Llama-2-7b-hf aya-expanse-8b aya-expanse-32b c4ai-command-r7b-12-2024 gemma-2-9b-it gemma-2-27b-it gemma-3-27b-it Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct\
    "

# --strategies TRAINING/LLAMA-2-7B/REVERSE/GEMMA/T=1.0/GREEDY TRAINING/LLAMA-2-7B/REVERSE/GEMMA/T=1.0/BS\
# --names checkpoint-200 checkpoint-400 checkpoint-600 checkpoint-800 checkpoint-1000 checkpoint-1200 checkpoint-1400 checkpoint-1600 checkpoint-1800 checkpoint-2000 checkpoint-2200 checkpoint-2400 checkpoint-2600 checkpoint-2800 checkpoint-3000 checkpoint-3200 checkpoint-3400 checkpoint-3600 checkpoint-3800 checkpoint-4000 checkpoint-4200 checkpoint-4400 checkpoint-4600 checkpoint-4800 checkpoint-5000\

ARGS="
    --model_name_or_path google/metricx-24-hybrid-xxl-v2p6\
    --dataset_name_or_path ntrex\
    --max_input_length 1024\
    --batch_size 1\
    --data_dir /path/to/GENERATIONS/NTREX\
    --number_of_predictions 1000\
    --seed 122\
    --num_workers 8\
    --metric metricx\
    --languages English Basque Hausa Igbo Kinyarwanda Nepali Somali Sundanese Swahili Urdu Xhosa\
    --strategies BENCHMARKING\
    --names FT/LLAMA-2-7B/MONO/BS/checkpoint-5000 FT/LLAMA-2-7B/MONO/GREEDY/checkpoint-5000 FT/LLAMA-3-8B/MONO/BS/checkpoint-5000 FT/LLAMA-3-8B/MONO/GREEDY/checkpoint-5000 FT/LLAMA-3-8B/MULTI/BS/checkpoint-100000 FT/LLAMA-3-8B/MULTI/GREEDY/checkpoint-100000 LLaMAX3-8B-Alpaca LLaMAX2-7B-Alpaca nllb-200-3.3B Meta-Llama-3.1-70B-Instruct Meta-Llama-3-8B Llama-2-7b-hf aya-expanse-8b aya-expanse-32b c4ai-command-r7b-12-2024 gemma-2-9b-it gemma-2-27b-it gemma-3-27b-it Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct\
    "


ARGS="
    --model_name_or_path google/metricx-24-hybrid-xxl-v2p6\
    --dataset_name_or_path tico\
    --max_input_length 1024\
    --batch_size 1\
    --data_dir /path/to/GENERATIONS/TICO\
    --number_of_predictions 1000\
    --seed 122\
    --num_workers 8\
    --metric metricx\
    --languages English Basque Hausa Igbo Kinyarwanda Nepali Somali Sundanese Swahili Urdu Xhosa\
    --strategies BENCHMARKING\
    --names FT/LLAMA-2-7B/MONO/BS/checkpoint-5000 FT/LLAMA-2-7B/MONO/GREEDY/checkpoint-5000 FT/LLAMA-3-8B/MONO/BS/checkpoint-5000 FT/LLAMA-3-8B/MONO/GREEDY/checkpoint-5000 FT/LLAMA-3-8B/MULTI/BS/checkpoint-100000 FT/LLAMA-3-8B/MULTI/GREEDY/checkpoint-100000 LLaMAX3-8B-Alpaca LLaMAX2-7B-Alpaca nllb-200-3.3B Meta-Llama-3.1-70B-Instruct Meta-Llama-3-8B Llama-2-7b-hf aya-expanse-8b aya-expanse-32b c4ai-command-r7b-12-2024 gemma-2-9b-it gemma-2-27b-it gemma-3-27b-it Llama-3.1-8B-Instruct Meta-Llama-3.1-70B-Instruct Qwen2.5-7B-Instruct Qwen2.5-32B-Instruct\
    "