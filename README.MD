# [EMNLP 2025 Main] SolEval: Benchmarking Large Language Models for Repository-Aware Solidity Smart Contract Generation

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

[contributors-shield]: https://img.shields.io/github/contributors/pzy2000/SolEval.svg?style=for-the-badge
[contributors-url]: https://github.com/pzy2000/SolEval/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/pzy2000/SolEval.svg?style=for-the-badge
[forks-url]: https://github.com/pzy2000/SolEval/network/members
[stars-shield]: https://img.shields.io/github/stars/pzy2000/SolEval.svg?style=for-the-badge
[stars-url]: https://github.com/pzy2000/SolEval/stargazers
[issues-shield]: https://img.shields.io/github/issues/pzy2000/SolEval.svg?style=for-the-badge
[issues-url]: https://github.com/pzy2000/SolEval/issues
[license-shield]: https://img.shields.io/github/license/pzy2000/SolEval.svg?style=for-the-badge&label=License
[license-url]: https://github.com/pzy2000/SolEval/blob/main/LICENSE

📄 Paper: [https://arxiv.org/abs/2502.18793](https://arxiv.org/abs/2502.18793) <!-- Replace when available -->

🏆 Leaderboard: Coming soon

📫 Contact: [Zhiyuan Peng](mailto:pzy2000@sjtu.edu.cn), [Xin Yin](mailto:xyin@zju.edu.cn)

# 📰 News

* \<2025/02> 🚀 SolEval repository open-sourced.
* \<2025/08> 🎉 SolEval accepted to EMNLP 2025 Main Conference.
<!-- * \<2025/MM/DD> 🚀 SolEval leaderboard online. -->

# 🚀 Quickstart

## Requirements

* **Operating System**: Linux / Windows (macOS not thoroughly tested but likely works).
* **Python**: 3.8+ (3.10/3.11 recommended).
* **Solidity toolchain**: Foundry (**forge**) and `solc` (use `solc-select` to manage versions).
* **Security analyzer**: **Slither** (Python 3.8+).
* *(Optional)* **LLMs**: GPT-4o, DeepSeek, Qwen, CodeLlama, OpenCode, etc.

## Installation

```bash
git clone https://github.com/pzy2000/SolEval.git
cd SolEval
pip install -r requirements.txt
```

Before running evaluations, prepare repository data and dependency metadata.

### Repositories

Download the original repositories and extract them under the project root:

* Download: [repository.zip](https://media.githubusercontent.com/media/pzy2000/SolEval/refs/heads/master/repository.zip)

After extraction, you should have `SolEval/repository`. This folder contains **28** subfolders (each corresponding to a GitHub project). **Do not modify any internal file structures**, or the evaluation scripts may fail. The file structure should look like this after extraction:

<details>
<summary><strong>File Structure (click to expand)</strong></summary>

```text
SolEval/
├── LICENSE
├── README.MD
├── data/
├── environment.txt
├── forge/
├── libtree-sitter-solidity.so
├── prebuilt/
├── repository/        # contains 28 subfolders (GitHub projects)
├── requirement.txt
├── run_forge_test.sh
├── run_patch_gen.sh
├── run_slither.sh
└── tools/
</details>


## Precheck

### Install Foundry (forge 0.2.0 2024-12-08 version)

Install from the [Foundry releases page](https://github.com/foundry-rs/foundry/releases) (or via the official install script). Note that using newer version will cause inconsistent behaviors (e.g., forge test errors of the repository).

### Install Slither

> **Note**
> Slither requires Python 3.8+. If you’re **not** using a [supported compilation framework](https://github.com/crytic/crytic-compile), you’ll need the Solidity compiler `solc`; we recommend managing versions with [solc-select](https://github.com/crytic/solc-select).

```bash
python3 -m pip install slither-analyzer
```

# 🧪 Evaluation

SolEval provides an end-to-end pipeline for **repository-aware** contract generation: patch generation → functional correctness (Pass\@k) → security (Slither) → gas (forge).

## 1) Patch Generation (LLM Reasoning)

Generate repository-aware patches with various LLMs; supports RAG and few-shot configurations.

```bash
# Example: GPT-4o + RAG, 1-shot, sampling 10 candidates per requirement
python generate_rag.py --context --model gpt-4o --shot 1 --sample 10

# Other configurations (uncomment and adjust as needed)
# python generate_rag.py --context --model gpt-4o-mini --shot 3 --sample 10
# python generate_rag.py --context --model Qwen-7B --shot 2 --sample 5
# python generate_random.py --context --model Qwen-7B --shot 1
# python generate_random.py --context --model OpenCode-33B --shot 2
```

**Args:**

* `--context`: enable repository context (e.g., RAG/retrieval).
* `--model`: LLM name used for generation.
* `--shot`: number of few-shot exemplars.
* `--sample`: number of candidates per requirement.

## 2) Pass\@k Evaluation (Functional Correctness)

Run tests via Foundry `forge` to evaluate functional correctness of generated patches.

```bash
python run_forge.py --context y --model DeepSeek-V3 --sample 1 --rag true --shot 1
python run_forge.py --context n --model DeepSeek-V3 --sample 1 --rag true --shot 1
python run_forge.py --context y --model DeepSeek-V3 --sample 1 --rag false --shot 1
python run_forge.py --context n --model DeepSeek-V3 --sample 1 --rag false --shot 1
```

**Args:**

* `context`: `y|n` to toggle repository context.
* `model`: model used for generation.
* `shot`: number of few-shot exemplars.
* `sample`: number of candidates per requirement.
* `rag`: whether to enable RAG.

## 3) Vulnerability Analysis (Slither)

Run static analysis for potential security issues using **Slither**. Requires the `verifier` file produced in the previous step.

```bash
python run_slither.py --context y --verifier results/rag/results_OpenCode_shot_1_context_True_testcase_False_20250130_033003.jsonl --model OpenCode --sample 10 --rag true
python run_slither.py --context y --verifier results/rag/results_DeepSeek-Coder-33B_shot_1_context_True_testcase_False_20250201_025654.jsonl --model DeepSeek-Coder-33B --sample 10 --rag true
python run_slither.py --context y --verifier results/rag/results_CodeLlama-34B_shot_1_context_True_testcase_False_20250201_064732.jsonl --model CodeLlama-34B --sample 10 --rag true
```

**Args:**

* `context`: `y|n` for repository context.
* `model`: generation model.
* `verifier`: path to the `.jsonl` verifier from the “Pass\@k Evaluation” step.
* `sample`: candidates per requirement.
* `rag`: whether to enable RAG.

## 4) Gas Analysis (forge)

Compute gas metrics using `forge`. First, place all `results_*.jsonl` under `results/gas`, then run **in order**:

```bash
python tools/utils/intersect_gas.py
python tools/run_gas.py --context y --model OpenCode --sample 10 --rag true --shot 1
# python tools/run_gas.py --context y --model DeepSeek-Coder-33B --sample 10 --rag true --shot 1
# python tools/run_gas.py --context y --model CodeLlama-34B --sample 10 --rag true --shot 1
```

**Args:**

* `context`: `y|n`.
* `model`: model name.
* `shot`: number of few-shot exemplars.
* `sample`: candidates per requirement.
* `rag`: whether to enable RAG.

> Outputs for functional correctness, security, and gas are written to `results/` (or to the directory specified by your scripts).

# ❓ Common Issues Checklist

Before opening an issue, please verify:

* [ ] `repository/` directory matches the provided originals (no internal paths/files altered).
* [ ] **Foundry**, `solc`/`solc-select`, and **Slither** are installed and callable from your shell.
* [ ] Python deps installed: `pip install -r requirements.txt`.
* [ ] Script parameters (`--model`/`--shot`/`--sample`/`--context`/`--rag`) match your expected setup.
* [ ] `verifier` path (for Slither) comes from the real output of the Pass\@k step.
* [ ] For gas analysis, `results_*.jsonl` are in `results/gas` and scripts executed in order.

# 👋 Overview

**SolEval** is an evaluation framework for **repository-aware** Solidity contract generation, targeting end-to-end development capabilities and engineering utility under realistic repository context. The framework covers patch generation, functional correctness (Pass\@k), security (Slither), and gas evaluation, and supports RAG and few-shot configurations for reproducible, apples-to-apples comparisons across LLMs/agents.

## Structure

* **Datasets / Repositories**: 9 real GitHub projects (original structure preserved).
* **Generation**: Multi-model, multi-sample patch generation with optional RAG/few-shot.
* **Testing**: Functional correctness via Foundry `forge` (Pass\@k).
* **Security**: Static analysis via Slither.
* **Gas**: Gas metrics via `forge`.
* **Artifacts**: Unified `results/*.jsonl` outputs for analysis and reproducibility.


# 🖊 Citation

```bibtex
@article{peng2025soleval,
  title={SolEval: Benchmarking Large Language Models for Repository-level Solidity Code Generation},
  author={Peng, Zhiyuan and Yin, Xin and Qian, Rui and Lin, Peiqin and Liu, Yongkang and Ying, Chenhao and Luo, Yuan},
  journal={arXiv preprint arXiv:2502.18793},
  year={2025}
}
```

# ❗ Known Issues

- [ ] Some repositories require specific `solc` versions; align with `solc-select` before running evaluations.
- [ ] On Windows, certain dependency installers may require Administrator privileges.

# ☘ Feature in Future

- [ ] Public leaderboard and full evaluation report.
- [ ] More real repositories and test cases.
- [ ] Additional languages/frameworks (e.g., Vyper / Hardhat).

# 🤝 Contributing

Contributions are welcome!

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push: `git push origin feature-branch`
5. Open a Pull Request.

Please follow the existing code style and include appropriate tests.

# 📄 License

Released under the **MIT License**. See [LICENSE](LICENSE) for details.

# 🙏 Acknowledgements

This repository accompanies our EMNLP 2025 Main paper. We thank the open-source community and the authors of tools used here (Foundry, Slither, `solc`, etc.).
