# VLM4JD: Vision-Language-Model for Jailbreak Defense

## Contents
- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Citation](#citation)
- [License](#license)

## Overview
Large Language Models (LLMs) have demonstrated exceptional utility but are vulnerable to adversarial jailbreak attacks, where carefully crafted prompts bypass safety mechanisms, causing LLMs to generate harmful content. Existing defense mechanisms, including supervised fine-tuning, preference optimization, and pre-stage LLM-based detectors, struggle against advanced jailbreaks that use iterative adversarial optimization. These attacks generate patterns that distract LLMs, leading to high attack success rates.

This research highlights a key observation: current text-based jailbreaks exhibit limited generalization when transferred to other modalities, such as text-embedded images. Based on this, we propose a novel jailbreak detection method named **Vision-Language-Model for Jailbreak Defense (VLM4JD)**. VLM4JD encodes textual prompts into visual signals and leverages the multimodal understanding capabilities of Vision-Language Models (VLMs) to detect harmful intent.

VLM4JD is lightweight, training-free, and highly effective. Our empirical evaluations on large jailbreak datasets (JailbreakBench, HarmBench) demonstrate that VLM4JD significantly reduces Attack Success Rates (ASR), outperforming traditional text-based detectors. This work uncovers the cross-modal generalization limitations of current jailbreak attacks and offers VLM4JD as a robust defense, aiming to enhance the understanding and mitigation of such vulnerabilities.

## Repo Contents
- **`main.py`**: The main script to run jailbreak detection experiments using VLM4JD and other baseline defenses.
- **`utils.py`**: Contains utility functions for processing prompts, interacting with models, and evaluating outputs (e.g., judging harmfulness).
- **`defences.py`**: Implements the VLM4JD defense mechanism along with other comparative defense strategies (e.g., SmoothLLM, Guard Model, Self-Eval).
- **`requirements.txt`**: (Recommended) A file listing all Python dependencies.
- **`LICENSE`**: The license for this project.

## System Requirements

### Hardware Requirements
* **GPU:** We run all experiments using a node with 8 * A100 80G
* **CPU:** AMD EPYC 7k62

### Software Requirements
* **Operating System:** Ubuntu 22.04
* **Python:** Python 3.10 or newer
* **Key Python Libraries:**
    * `torch`
    * `transformers`
    * `vllm`
    * `jailbreakbench`
    * `scikit-learn`

## Installation Guide

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/xiyuanyang45/VLM-for-jailbreak-defense
    cd VLM4JD
    ```

2.  **Set up a Python virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install Python dependencies:**
    It's recommended to create a `requirements.txt` file with all necessary packages. If you have one:
    ```bash
    pip install -r requirements.txt
    ```
    Alternatively, install the core packages manually:
    ```bash
    pip install torch transformers vllm jailbreakbench scikit-learn Pillow
    ```

4.  **Hugging Face Hub:**
    If you plan to use models that require authentication (e.g., some Llama models), log in using the Hugging Face CLI:
    ```bash
    huggingface-cli login
    ```

5.  **Datasets:**
    This project utilizes datasets like JailbreakBench and HarmBench. The `jailbreakbench` library typically handles the download and management of its standard benchmarks.
    The `get_all_prompts` function in `utils.py` (or similar) should interface with `jailbreakbench` to load adversarial prompts.
    For the `xstest` dataset used for over-rejection tests (`get_overrej_test_prompts`), ensure the dataset is available in the expected path or provide instructions if manual download/setup is needed.

## Demo

The main script `main.py` is used to run experiments. Below are some examples:

### 1. Running VLM4JD Defense
To run VLM4JD defense against GCG attacks on Vicuna-13b-v1.5 using Phi-3.5-vision as the VLM detector on the JailbreakBench dataset:
```bash
python main.py \
    --method GCG \
    --model_been_attack vicuna-13b-v1.5 \
    --vlm microsoft/Phi-3.5-vision-instruct \
    --defences vlm \
    --dataset jailbreakbench
```

You can specify multiple VLMs for an ensemble defense by comma-separating them in the `--vlm` argument:

```bash
python main.py \
    --method GCG \
    --model_been_attack vicuna-13b-v1.5 \
    --vlm "microsoft/Phi-3.5-vision-instruct,Qwen/Qwen2-VL-2B-Instruct" \
    --defences vlm \
    --dataset jailbreakbench
```

### 2\. Running Without Any Defense (Baseline)

To evaluate the attack success rate without any defense:

```bash
python main.py \
    --method GCG \
    --model_been_attack vicuna-13b-v1.5 \
    --wo_defence 1 \
    --dataset jailbreakbench
```

### 3\. Testing Other Defenses

To test SmoothLLM defense (variant 1):

```bash
python main.py \
    --method GCG \
    --model_been_attack vicuna-13b-v1.5 \
    --defences smoothllm \
    --smooth_var 1 \
    --dataset jailbreakbench
```

To test Llama Guard 3.1B as a defense:

```bash
python main.py \
    --method GCG \
    --model_been_attack vicuna-13b-v1.5 \
    --defences guard_model \
    --guard_model meta-llama/Llama-Guard-3-1B \
    --dataset jailbreakbench
```

### 4\. Testing Over-Rejection

To test the over-rejection rate of VLM4JD on the `xstest` dataset (assuming `xstest` contains benign prompts):

```bash
python main.py \
    --test_over_rej 1 \
    --over_rej_dataset xstest \
    --defences vlm \
    --vlm microsoft/Phi-3.5-vision-instruct
```

### Key Arguments:

  * `--method`: Adversarial attack method (e.g., `GCG`).
  * `--model_been_attack`: The target LLM being attacked (e.g., `vicuna-13b-v1.5`, `llama-2-7b-chat-hf`).
  * `--vlm`: The VLM(s) used for the VLM4JD defense. Can be a single model or comma-separated for ensemble. Shorthand names (e.g., `phi3.5`) or full Hugging Face paths are supported.
  * `--defences`: Comma-separated list of defense methods to apply (e.g., `vlm`, `smoothllm`, `guard_model`, `self_eval`, `llm_eval`, `vlm_text`).
  * `--wo_defence`: Set to `1` to run without any defense.
  * `--test_over_rej`: Set to `1` to test over-rejection on benign prompts.
  * `--over_rej_dataset`: Specifies the dataset for over-rejection testing (e.g., `xstest`).
  * `--dataset`: Specifies the primary jailbreak dataset (e.g., `jailbreakbench`, `harmbench`).
  * `--smooth_var`: Variant for SmoothLLM defense.
  * `--guard_model`: Path to the guard model for `guard_model` defense.
  * `--llm_eval`: Path to the LLM used for `llm_eval` defense.
  * `--vlm_text`: Path to the VLM used for `vlm_text` (text-only modality of VLM) defense.


## Citation

Comming soon.

## License

This project is licensed under the terms of the [LICENSE](https://github.com/xiyuanyang45/VLM-for-jailbreak-defense/blob/main/LICENSE) file.
