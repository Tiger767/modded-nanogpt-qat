# Project Proposal

Student: Travis Hammond

## Project Idea: Low-Bit Quantized nanoGPT Comparison

My project is to implement and evaluate a **low-bit quantized Large Language Model (LLM)** based on the GPT-style architecture, using the "modded-nanogpt" codebase as a foundation. The core of the project will be a direct comparison between the baseline full-precision model and its 1.58-bit (ternary) quantized versions, inspired by the BitNet research.

The project will focus primarily on weight-only quantization, with activation quantization as a secondary exploration:

1.  **Primary Focus (W1.58A_BF16):** Constraining model weights to $\{-1, 0, 1\}$ while keeping activations in high precision (e.g., BF16). This will isolate the impact of weight-only quantization.
2.  **Secondary Exploration (W1.58A_8/4):** If the primary model shows sufficient performance, I will explore further constraining activations to low-bit precision (e.g., 8-bit or 4-bit), following the full BitNet b1.58 scheme.

This investigation aims to quantify the trade-offs between model compression, computational efficiency, and performance in a practical, small-scale setting.

***

## Expected Differences and Significance

The implemented quantized models are expected to differ significantly from the base `modded-nanogpt` model (which typically uses FP32 or BF16) in the following ways:

* **Memory Footprint:** The 1.58-bit weights should drastically reduce the memory consumption for storing the weights, leading to much lower bandwidth requirements.

* **Computational Efficiency:** The ternary weights enable a new computation paradigm, allowing matrix multiplication to be replaced largely by integer addition (specifically, addition-only for -1/+1 and skip-if-zero for 0). This eliminates the need for
    costly floating-point multiplication, a major factor in energy consumption.

* **Convergence and Performance:** The low-bit models (especially W1.58A_BF16) will be compared directly to the full-precision baseline. While a
    slight performance loss is often expected, the 1.58-bit quantization may act as a strong regularizer. It will be interesting to observe if this effect allows the quantized model to **match or even exceed** the baseline's performance (e.g., lower perplexity) on the given dataset. The secondary `W1.58A_8/4` model is expected to show a more significant performance trade-off in exchange for even greater efficiency.

***

## Specific Experiments and Key Metrics

The evaluation will be centered around a comparative analysis between the following models, trained from scratch using Quantization-Aware Training (QAT) techniques (e.g., Straight-Through Estimator):

1.  **Baseline:** Full-precision `modded-nanogpt` model (BF16/FP32 weights and activations).
2.  **Primary Quantized Model:** **W1.58A_BF16** (Ternary weights, BF16 activations). This is the main focus.
3.  **Secondary Quantized Model (Conditional):** **W1.58A_8** (or A_4) QAT model (Ternary weights, low-bit activations). This will be pursued if the primary model's performance is sufficient.

### Key Metrics for Comparison:

* **Model Performance:**
    * **Validation Loss / Perplexity (PPL):** To measure the quality of the language model's predictions and convergence.
    * **Downstream Task Accuracy (Optional, scale permitting):** If training allows, a simple zero-shot task (e.g., BoolQ or Hellaswag) will be evaluated.

* **Efficiency and Cost:**
    * **Weight Memory Footprint (in bytes):** To quantify the reduction in model size.
    * **Training Time / Time Per Iteration (sec):** To see how the quantized forward/backward pass impacts training speed.
    * **Training Energy/Ops Reduction Estimate (Theoretical):** To estimate the arithmetic operation energy cost reduction based on theoretical ratios.

* **Core Comparison:**
    * **Convergence Rate and Final Loss/PPL:** Directly comparing the performance curves of the **Baseline** vs. the **Primary Quantized Model** to assess the viability of 1.58-bit weight quantization.

#### Datasets and Libraries:

* **Codebase:** A fork of "modded-nanogpt" to be adapted for QAT.
* **Dataset:** A modest-sized text corpus (FineWeb) to ensure models are large enough for non-trivial training yet small enough for practical experimentation on limited hardware.

***

## Optional Exploration: External Knowledge Tokens

If time allows after the primary quantization comparisons are complete, I will explore a novel hybrid approach. This involves injecting a small set of **"External Knowledge Tokens"** into the model's attention mechanism.

* **Concept:** Unlike the model's standard quantized parameters, these would be **actual tokens (vectors) maintained in full precision (BF16/FP32)**. They would be prepended to the input sequence and processed by the attention layers.
* **Hypothesis:** These full-precision tokens could act as a small, high-capacity "scratchpad" or knowledge store. This might allow the model to offload critical information (e.g., complex factual mappings or reasoning steps) that is difficult to represent or process using only the low-bit, ternary weight space.
* **Experiment:** The experiment would compare the **Primary Quantized Model (W1.58A_BF16)** against a hybrid version (W1.58A_BF16 + Knowledge Tokens) to see if this addition provides a measurable boost in performance (e.g., lower perplexity) with a minimal, fixed increase in parameter count.

The project will focus on the most memory-intensive components—the weights and activations—and introduce a novel structural modification to explore ways to mitigate the inherent information bottleneck of extreme quantization, with an optional exploration into alternative low-memory training algorithms.

***

## References

1.  Cabral, E. L. L., Pirozelli, P., & Driemeier, L. (2025). *1 BIT IS ALL WE NEED: BINARY NORMALIZED NEURAL NETWORKS*. arXiv preprint arXiv:2509.07025.
2.  Semenov, S. (2025). *Smooth Approximations of the Rounding Function*. arXiv preprint arXiv:2504.19026.
3.  Daliri, M., Song, Z., & Yang, C. (2024). *Unlocking the Theory Behind Scaling 1-Bit Neural Networks*. arXiv preprint arXiv:2411.01663.
4.  Zhao, K., Tabaru, T., Kobayashi, K., Honda, T., Yamazaki, M., & Tsuruoka, Y. (2025). *Direct Quantized Training of Language Models with Stochastic Rounding*. Proceedings of Machine Learning Research 304 (ACML 2025). arXiv preprint arXiv:2412.04787.
5.  Li, H., Hong, J., Adbol, S., Wu, Y., & Li, Z. (2024). *Continuous Approximations for Improving Quantization Aware Training of LLMs*. arXiv preprint arXiv:2410.10849.
6.  Ma, S., Wang, H., Ma, L., Wang, L., Wang, W., Huang, S., Dong, L., Wang, R., Xue, J., & Wei, F. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. arXiv preprint arXiv:2402.17764.
7.  Wang, H., Ma, S., Dong, L., Huang, S., Wang, H., Ma, L., Yang, F., Wang, R., Wu, Y., & Wei, F. (2023). *BitNet: Scaling 1-bit Transformers for Large Language Models*. arXiv preprint arXiv:2310.11453.
8.  Tabesh, R., Safaryan, M., & Alistarh, D. (2025). *CAGE: CURVATURE-AWARE GRADIENT ESTIMATION FOR ACCURATE QUANTIZATION-AWARE TRAINING*. arXiv preprint arXiv:2510.18784.
9.  Brickner, W. (2024). *NOISE_STEP: TRAINING IN 1.58B WITH NO GRADIENT MEMORY*.
