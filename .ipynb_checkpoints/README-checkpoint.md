# Why Alignment Should Precede Knowledge Distillation

This repository contains the official implementation for the paper: **"Why Alignment Should Precede Knowledge Distillation"** (NeurIPS 2025).

**Authors:** Sungmin Cha and Kyunghyun Cho

---

### 📖 Abstract

Optimizing Large Language Models (LLMs) for deployment typically involves two critical post-training stages: Alignment (e.g., RLHF, DPO) to ensure performance for target tasks, and Knowledge Distillation (KD) to reduce computational costs. However, the optimal scheduling of these two processes remains an open question.

In this work, we investigate the interaction between alignment and distillation, comparing two distinct pipelines:

1. Distill-then-Align (K→A): Distilling a pre-trained teacher into a small student model, followed by alignment on the student.

2. Align-then-Distill (A→K): Aligning the teacher model first, then distilling its aligned distribution into the small student model.

We provide a theoretical analysis and experimental validation using Mixture of Gaussians (MoG), RLHF (PPO), and DPO on LLMs. Our results demonstrate that Alignment should precede Knowledge Distillation. We show that the A→K pipeline allows the student to directly approximate the aligned policy's high-probability regions, whereas K→A forces the capacity-limited student to navigate the optimization landscape of alignment from a suboptimal initialization.

---

### ⚙️ Setup

Please install the necessary dependencies before running the experiments.

```bash
pip install -r requirements.txt

```

---

### 1. Mixture of Gaussians (MoG) Experiment

We provide a self-contained experiment to demonstrate the theoretical properties using a controlled Mixture of Gaussians (MoG) environment.

* **Directory:** `MoG/`
* All codes related to the MoG experiments are located in the `./MoG/` directory.

---

### 2. Model Preparation

Before running the main experiments, please download the necessary model checkpoints.

1. Download the **Pretrained** and **Target** models from the following link:
* **[Insert Link Here]**


2. After downloading, place the `pretrained_models/` and `target_models/` directories into the root of this repository (`./why-alignment-precede-kd/`).

The directory structure should look like this:

```text
why-alignment-precede-kd/
├── pretrained_models/
├── target_models/
├── MoG/
├── scripts...

```

---

### 3. Experimental Pipeline

We compare two distinct pipelines: **KD  Alignment** and **Alignment  KD**.

#### 3.1 Scenario A: KD  Alignment (KD-First)

In this setting, we first distill the model and then apply alignment techniques.

**RLHF (PPO)**
To run the experiment using PPO:

```bash
bash run_rlhf_ppo_k_a.sh

```

**DPO (Direct Preference Optimization)**
To run the experiment using Online or Offline DPO:

```bash
# Online DPO
bash run_dpo_online_k_a.sh

# Offline DPO
bash run_dpo_offline_k_a.sh

```

---

#### 3.2 Scenario B: Alignment  KD (Alignment-First)

In this setting, we align the teacher model first, followed by knowledge distillation to the student.

**Step 1: Alignment**
First, perform alignment on the model using one of the following methods:

* **RLHF (PPO):**
```bash
bash run_rlhf_ppo_a_k.sh

```


* **DPO (Online/Offline):**
```bash
# Online DPO
bash run_dpo_online_a_k.sh

# Offline DPO
bash run_dpo_offline_a_k.sh

```



**Step 2: Knowledge Distillation**
After the alignment phase is complete (Step 1), proceed with the distillation process:

1. **Generate Data:** Generate the dataset using the aligned model.
```bash
bash run_generate_data_for_kd.sh

```


2. **Train Student Model:** Train the 135M student model (p'') using the generated data.
```bash
bash run_kd.sh

```



---

### 🔗 Citation

If you find this work or code useful, please cite our paper:

```bibtex
@article{cha2025alignment,
  title={Why Alignment Must Precede Distillation: A Minimal Working Explanation},
  author={Cha, Sungmin and Cho, Kyunghyun},
  journal={arXiv preprint arXiv:2509.23667},
  year={2025}
}

```