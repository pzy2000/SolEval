# SolEval: Benchmarking Large Language Models for Repository-level Solidity Smart Contract Generation

**SolEval** is an evaluation framework designed to benchmark Large Language Models (LLMs) for generating Solidity smart contracts at the repository level.

[//]: # (## Features)

[//]: # ()
[//]: # (- **Automated Smart Contract Generation**)

[//]: # (- **Repository-Level Evaluation**)

[//]: # (- **Security Vulnerability Detection**)

[//]: # (- **Test Execution**)

## Installation

To install SolEval, clone this repository and install the required dependencies:

```bash
pip install -r requirements.txt
```

Before running the evaluation, researchers need to download the repositories, and dependency data.

### Repositories

The original repositories can be downloaded from [link](https://anonymous.4open.science/r/SolEval-1C06/repository.zip). Users need to uncompressed the repositories and put them in the root directory (i.e., `SolEval/repository`).

The project are stored in `repository`. `repository` contains 9 subfolders, each of which corresponds to a GitHub project. Please do not modify the file structure of the repositories. Otherwise, the evaluation script can not work properly.


## Precheck

Before running the evaluation, ensure that you have installed the following tools:

### How to install Forge
Precompiled binaries can be downloaded from the [GitHub releases page](https://github.com/foundry-rs/foundry/releases). 


### How to install Slither
> **Note** <br />
> Slither requires Python 3.8+.
If you're **not** going to use one of the [supported compilation frameworks](https://github.com/crytic/crytic-compile), you need [solc](https://github.com/ethereum/solidity/), the Solidity compiler; we recommend using [solc-select](https://github.com/crytic/solc-select) to conveniently switch between solc versions.

```console
python3 -m pip install slither-analyzer
```

## Usage

### Generating Patches with LLMs

You can generate patches for Solidity contracts using the `run_patch_gen.sh` script. This script calls various LLMs to generate code patches with different configurations:

```bash
# Example command for generating a patch with GPT-4o
python generate_rag.py --context --model gpt-4o --shot 1 --sample 10

# Uncomment and adjust the lines below for other models and configurations
# python generate_rag.py --context --model gpt-4o-mini --shot 3 --sample 10
# python generate_rag.py --context --model Qwen-7B --shot 2 --sample 5
# python generate_random.py --context --model Qwen-7B --shot 1
# python generate_random.py --context --model OpenCode-33B --shot 2
```

### Running Pass@k Evaluation (Functional Correctness)

To run tests on the generated contracts, use the `run_forge_test.sh` script. This will run tests using `forge` with different configurations for context, sample size, and whether or not to use RAG:

```bash
python run_forge.py --context y --model DeepSeek-V3 --sample 1 --rag true --shot 1
python run_forge.py --context n --model DeepSeek-V3 --sample 1 --rag true --shot 1
python run_forge.py --context y --model DeepSeek-V3 --sample 1 --rag false --shot 1
python run_forge.py --context n --model DeepSeek-V3 --sample 1 --rag false --shot 1
```

The arguments are explained as follows.
- `context`: y or n, whether to use context or not.
- `model`: the LLM used for generating the contracts.
- `shot`: the number of shots for prompt-based generation.
- `sample`: the number of generated programs per requirement.

### Running Vulnerability Analysis with Slither

You can run vulnerability checks using Slither with the `run_slither.sh` script. This will analyze the Solidity contract for potential security vulnerabilities. The script takes in a verifier file generated from the second step and runs the analysis:

```bash
python run_slither.py --context y --verifier results/rag/results_OpenCode_shot_1_context_True_testcase_False_20250130_033003.jsonl --model OpenCode --sample 10 --rag true
python run_slither.py --context y --verifier results/rag/results_DeepSeek-Coder-33B_shot_1_context_True_testcase_False_20250201_025654.jsonl --model DeepSeek-Coder-33B --sample 10 --rag true
python run_slither.py --context y --verifier results/rag/results_CodeLlama-34B_shot_1_context_True_testcase_False_20250201_064732.jsonl --model CodeLlama-34B --sample 10 --rag true
```

The arguments are explained as follows.
- `context`: y or n, whether to use context or not.
- `model`: the LLM used for generating the contracts.
- `verifier`: the verifier file generated from the `Running Pass@k Evaluation` step.
- `sample`: the number of generated programs per requirement.
- `rag`: whether to use RAG or not.


### Running Gas Analysis with forge

You can run gas analysis using forge with the `tools/run_gas.py` script. First, you need to put all the `results_*.jsonl` in the `results/gas` folder. Then, you can run the following command SEQUENTIALLY:

```bash
python tools/utils/intersect_gas.py
python tools/run_gas.py --context y --model OpenCode --sample 10 --rag true --shot 1
#python tools/run_gas.py --context y --model DeepSeek-Coder-33B --sample 10 --rag true --shot 1
#python tools/run_gas.py --context y --model CodeLlama-34B --sample 10 --rag true --shot 1
```

The arguments are explained as follows.
- `context`: y or n, whether to use context or not.
- `model`: the LLM used for generating the contracts.
- `shot`: the number of shots.
- `sample`: the number of generated programs per requirement.
- `rag`: whether to use RAG or not.

## Contributing

We welcome contributions to SolEval. To contribute:

1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

Please ensure your code adheres to the existing coding standards and includes appropriate tests.

## License

SolEval is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

SolEval is developed as part of our submission for EMNLP2025.

We acknowledge the contributions of the open-source community and the developers of the tools integrated into SolEval.

## Contact

For questions or feedback, please open an issue on this repository.
