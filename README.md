# Low-Bit Quantized nanoGPT Speedrun (QAT Research)

**Student:** Travis Hammond
**Course Project:** QAT Research focusing on 1.58-bit Quantization

This repository implements a comparative study of the [modded-nanogpt project](https://github.com/KellerJordan/modded-nanogpt/tree/master) against various low-bit quantized models, inspired by BitNet's ternary ($W_{1.58}$) quantization. The project focuses on isolating the effects of weight-only quantization versus combined weight-and-activation quantization on training stability, convergence, and final validation loss.

## Project Structure and Key Files

| File/Folder | Description |
| :--- | :--- |
| `configs/` | Contains JSON configuration files that define the exact quantization flags used for each experiment. This ensures reproducibility. |
| `logs/` | Stores training log files (`.txt`) and model checkpoints (`.pt`) for each completed run. Logs are categorized into `normal` and `quantization` folders. |
| `train_gpt.py` | The main training script. It is modified to read the quantization settings from a specified configuration file. |
| `sample.py` | The model inference/generation script. This acts as the project's primary "main.py" equivalent. It must be run with the checkpoint and the matching configuration file to correctly rebuild the quantized model architecture. |

## Installation

The project was primarily developed on a single NVIDIA H100 GPU.

1.  **Clone and Setup Venv:**

    ```bash
    git clone https://github.com/Tiger767/modded-nanogpt-lowbit.git && cd modded-nanogpt-lowbit
    python3 -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies and PyTorch:**

    ```bash
    pip install -r requirements.txt
    sudo pip uninstall torch
    # Ensure a version compatible with CUDA 12.8 is installed
    pip install torch --index-url https://download.pytorch.org/whl/cu128
    ```

3.  **Prepare Data (FineWeb):**

    ```bash
    python data/cached_fineweb10B.py 9
    ```

### Docker Alternative Setup

```bash
sudo docker build -t modded-nanogpt .
sudo docker run -it --rm --gpus all -v $(pwd):/modded-nanogpt modded-nanogpt bash
```

## Usage

### Training (`train_gpt.py`)

To train a model, specify the desired configuration file. This will write a new log and checkpoint to the `logs/` directory.

```bash
# Example: Training with the Weights-Only quantization config
torchrun --standalone --nproc_per_node=1 train_gpt.py --config configs/quant_weights.json  # change nproc_per_node to 1 for single H100
```

### Inference (`sample.py`)

To ensure the model is loaded with the correct low-bit architecture, the checkpoint must be paired with its specific training configuration file.

```bash
# Example: Generating text using the Extreme Quantization checkpoint
python sample.py \
  --checkpoint logs/quantization/7445d176-c123-4058-86c9-17a9d57fdcc0.state_step002285.pt \
  --config configs/quant_extreme.json \
  --prompt "The meaning of life is"
```

## Experiments and Results

Four key experiments were conducted, focusing on 1.58-bit (ternary) weight quantization and various levels of activation quantization.

| Experiment Name | Log/Checkpoint ID | Configuration File | Final Val Loss | Step Avg Time (1xH100) |
| :--- | :--- | :--- | :--- | :--- |
| **1. Normal Baseline** | `d82fe2fe-b7be-4bc1-b390-49e6650043d5` | `configs/normal.json` | **3.2773** | 706.65ms |
| **2. Weights Only** | `da281a7d-8d00-4603-bc80-2c85c753dc25` | `configs/quant_weights.json` | 3.4105 | 726.32ms |
| **3. Weights + Attn Act** | `63f9cd47-8dea-4687-b648-23a727d21da5` | `configs/quant_weights_attn_act.json` | 3.4894 | 751.22ms |
| **4. Extreme Full Quant** | `7445d176-c123-4058-86c9-17a9d57fdcc0` | `configs/quant_extreme.json` | 3.8690 | 556.06ms (H100 SXM) |

## Micro-Ablation Studies

Prior to the full training runs, a series of short "micro-ablations" (400 steps) were conducted to screen different activation functions and quantization strategies.

| Ablation Variant | Val Loss @ Step 400 | Observation |
| :--- | :--- | :--- |
| **Full Precision (Control)** | **3.8990** | Baseline reference point. |
| **Quant Weights Only** | 4.0344 | Minimal degradation; accepted for full run. |
| **Quant Weights + 1.58b Tanh QK Act** | 4.1286 | Reasonable trade-off; accepted for Exp 3. |
| **All 8-bit Activations (ReLU2)** | 4.1716 | 8-bit activations are viable but seem useless in light of FP8 methods. |
| **All Tanh Activations** | 4.4001 | Tanh (1.58-bit) acts converge far better compared to sigmoid activations. |
| **All Sigmoid Activations** | 4.8493 | Sigmoid (1-bit) activations caused severe signal decay. |

## Inference and Efficiency Implications

The primary motivation for $W_{1.58}$ quantization is achieving **massive model compression and increased computational efficiency** during inference.

  * **Model Size and Memory (Weights Only - Exp 2):** By constraining weights to the ternary values $\{-1, 0, 1\}$, the memory footprint for storing model weights is drastically reduced, enabling **smaller models** and **faster loading times**. The minimal performance degradation ($\approx 0.13$ loss increase) compared to the baseline suggests this is a highly effective compression technique.
  * **Computational Speed (Ternary Arithmetic):** When both weights and activations are constrained to $\{-1, 0, 1\}$ (or another low-bit format), standard floating-point multiplication can be replaced largely by **integer addition** (and subtraction, skipping the zero term). This eliminates costly floating-point operations, leading to **significantly faster matrix multiplication** and lower energy consumption.
  * **Context Memory Overhead (Activation Quantization):** While challenging to train, successful low-bit quantization of activations, particularly the Query and Key (QK) vectors in the attention layers (as explored in Exp 3 and 4), offers the critical benefit of **reduced memory overhead for the KV-Cache** during inference. This allows the model to handle a **much larger context window** on limited hardware.

The results show a clear trade-off: **Experiment 2 (Weights Only)** provides the best balance, delivering the primary memory and speed benefits with minimal performance cost. **Experiment 4 (Extreme Full Quantization)**, while maximizing the theoretical computational benefits (faster matrix multiplication), confirms that sustaining high performance is difficult when activations are aggressively constrained, leading to severe performance collapse (Val Loss 3.8690).
