# Welcome to RL-Align

**Extreme infrastructure and specialized custom kernels for large-scale reinforcement learning.**

We are an open-source collective dedicated to solving the most critical bottlenecks in modern RL post-training pipelines like RLHF, GRPO, PPO. Our mission is to eliminate train-inference divergence and break the memory and compute walls, providing plug-and-play CUDA and Triton primitives that seamlessly integrate into your existing distributed frameworks like vLLM, vime, verl, Megatron, DeepSpeed etc.

### Our Core Philosophy

1. **Train-Inference Alignment:** The biggest hidden killer in large-scale RL is the subtle numerical divergence between rollout engines like vLLM FP8/BF16 and training engines like PyTorch native ops. We provide mathematically rigorous, fused operators that lock down the computational graph, guaranteeing absolute numerical consistency across the entire RL loop to prevent reward hacking and distribution drift.
2. **Extreme Memory & Compute Efficiency:** We replace naive O(G · L · V) native PyTorch paths with specialized kernels like prefix_shared_attention and fused_logp, reducing VRAM consumption by up to 10x and unlocking massive batch sizes for GRPO workloads without OOM.

## Core Projects

* [**RL-Kernel**](https://github.com/RL-Align/RL-Kernel): Our flagship project. A high-performance, mathematically rigorous kernel library providing fused primitives like fused_logp, grpo_loss, prefix_shared_attention designed to enforce train-inference alignment while aggressively optimizing memory and latency.
 
## Join the Community

Whether you are a kernel hacker, an AI Infra engineer, or an LLM researcher, you are welcome here!

* **Discord:** [Join our server](https://discord.gg/RGUQrr74z) for real-time technical discussions.
* **Contact:** Reach out to us at team@rl-align.org for collaborations.

---
