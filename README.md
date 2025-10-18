# arc_transduction
Testing specialization of model for transduction with ARC

To send .env to VM: scp -P XXXX .env user@YYYY:/path/to/destination/

To create .env in VM: echo "HF_TOKEN=" > .env
To create distant kernel uv run --active python -m ipykernel install --user --name project --display-name "Python (project)"

git clone https://github.com/axeld5/arc_back.git && cd arc_back
sudo snap install astral-uv --classic && sudo uv sync
sudo uv pip install unsloth unsloth-zoo
sudo uv pip install triton && sudo uv pip install kernels
sudo uv pip install openai-harmony
sudo uv pip install --force-reinstall vllm --torch-backend=auto

For finetuning
sudo uv run induction_data_prep.py && sudo uv run accelerate launch induction_sft.py
sudo uv run induction_eval.py
sudo uv run induction_rl.py "qwen2.5_7b_singled_out_sft/merged"

To serve vllm model for RL
CUDA_VISIBLE_DEVICES=0 sudo uv run vllm serve qwen3_4b_singled_out_sft/merged --tensor-parallel-size 1 --max-model-len 32768
export CUDA_VISIBLE_DEVICES=0,1 && sudo --preserve-env=CUDA_VISIBLE_DEVICES uv run trl vllm-serve --model qwen2.5_7b_singled_out_sft/merged --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --max-model-len 32768
export CUDA_VISIBLE_DEVICES=2,3 && sudo --preserve-env=CUDA_VISIBLE_DEVICES uv run accelerate launch  --num_processes=2  --mixed_precision=bf16 induction_rl.py

other potential case
export CUDA_VISIBLE_DEVICES=0,1 && sudo --preserve-env=CUDA_VISIBLE_DEVICES uv run trl vllm-serve --model julien31/Soar-qwen-7b --host 12
7.0.0.1 --port 8000 --tensor-parallel-size 1 --max-model-len 32768
export CUDA_VISIBLE_DEVICES=0,1 && sudo --preserve-env=CUDA_VISIBLE_DEVICES uv run trl vllm-serve --model qwen2.5_7b_induction_rl_partial/merged --host 127.0.0.1 --port 8000 --tensor-parallel-size 1 --max-model-len 32768