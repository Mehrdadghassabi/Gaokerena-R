<div align="center">
  <img width="300" height="300" alt="logo_with_bg" src="https://github.com/user-attachments/assets/e029d223-4e85-40a9-8310-436fa7c430fc" />
</div>

<p align="center">
📃 <a href="https://arxiv.org/pdf/2510.20059" target="_blank">Paper</a> ｜🤗 <a href="https://huggingface.co/gaokerena" target="_blank">huggingface repository</a>
</p>

## 📒 Table of Contents
- [📒 Table of Contents](#-table-of-contents)
- [📍 Overview](#-overview)
- [🏃 Training process](#-training-process)
- [📊 Results](#-Results)
- [⚠️ Risks and Limitations](#-risks-and-limitations)
- [⛔️ License](#-license)
- [🤝 Collaborators](#-collaborators)

---

## 📍 Overview
We present gaokerena-R, a model trained with a limited-data approach to enhance the Persian medical reasoning capabilities of the [aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b) model. Despite using less data, gaokerena-R outperforms our previous model, [gaokerena-V](https://github.com/Mehrdadghassabi/Gaokerena-V), which was trained on a much larger dataset. This demonstrates the effectiveness of our reasoning-focused training strategy under data-constrained conditions.

## 🏃 Training process
Two methods were proposed to enhance the reasoning capabilities of the baseline model.
In both approaches, a teacher model guides the baseline model using Direct Preference Optimization (DPO).
We primarily used Method A due to the time-consuming nature of Method B.

### Method A
In this method, a teacher model tries to correct the student model’s reasoning errors.

<img width="521" height="407" alt="fig1" src="https://github.com/user-attachments/assets/e998c3ac-8fb7-4fea-a59b-92fa69d30355" />

### Method B
In this method, a teacher model critiques the student’s answer and guides it through a conversation to reach the correct answer.

<img width="551" height="365" alt="fig2" src="https://github.com/user-attachments/assets/9866f85d-6102-4168-99e9-7b289f1ea9c5" />

## 📊 Results

|                       | gaokerena-R + aya-expanse-8b(verifier) | gaokerena-V | aya-expanse-8b |
|-----------------------|--------------------|---------------------------|---------|
| **MMLU-anatomy(fa)**  | 47.40           | **48.14**                    | 40.74   |
| **MMLU-medicalgenetics(fa)**      | **56.0**           | 53.0          | 49.0    |
| **MMLU-collegemedicine(fa)**      | **50.28**              | 43.93     | 44.51   |
| **MMLU-clinicalknowledge(fa)**     | **58.86**          | 55.47        | 52.07   | 
| **MMLU-professionalmedicine(fa)**  | **48.89**          | 47.05        | 45.58   |
| **MMLU-collegebiology(fa)**      | **54.86**          | 47.22          | 45.14   | 
| **MMLU(avg)**         | **52.98**          | 49.31                     | 46.64   | 
| **IBMSEE Sept 2023**   | **46.42**          | 38.69                    | 34.52   |
| **Prompt**         | COT for the main model & Straight for the verifier          | Straight                   | Straight   | 
| **Inference_time**         | $\approx 5 \times 35 + 10 + 8s $       |  $\approx 10s$     | $\approx 10s$  | 


## ⚠️ Risks and Limitations
While Gaokerena aims to provide relatively accurate information, it is not a substitute for professional medical advice. The model may have limitations in:

- Handling medical emergencies.
- Addressing highly specialized or rare medical conditions.
- Offering region-specific guidance, as the training data does not include localized Persian medical practices.

## ⛔️ License
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial use only)

## 🤝 Collaborators
1. Mehrdad Ghassabi
2. Sadra Hakim
3. Dr. Hamid Reza Baradaran Kashani
4. Pedram Rostami

