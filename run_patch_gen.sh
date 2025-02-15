export CUDA_VISIBLE_DEVICES=0
#python tools/generate_rag.py --context --model DeepSeek-V3 --shot 3 --sample 1
#python tools/generate_random.py --context --model DeepSeek-V3 --shot 3 --sample 1
#python tools/generate_rag.py --context --model gpt-4o-mini --shot 3 --sample 10
python tools/generate_rag.py --context --model CodeLlama_7b --shot 3 --sample 1
#python tools/generate_random.py --context --model Qwen-7B --shot 1
#python tools/generate_random.py --context --model OpenCode-33B --shot 2