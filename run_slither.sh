python tools/run_slither.py --context y --verifier results/rag/results_OpenCode_shot_1_context_True_testcase_False_20250130_033003.jsonl --model OpenCode --sample 10 --rag true
python tools/run_slither.py --context y --verifier results/rag/results_DeepSeek-Coder-33B_shot_1_context_True_testcase_False_20250201_025654.jsonl --model DeepSeek-Coder-33B --sample 10 --rag true
python tools/run_slither.py --context y --verifier results/rag/results_CodeLlama-34B_shot_1_context_True_testcase_False_20250201_064732.jsonl --model CodeLlama-34B --sample 10 --rag true
