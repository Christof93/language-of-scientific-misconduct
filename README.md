# 🧪 Analyzing the Evolution of Scientific Misconduct Based on the Language of Retracted Papers

This repository contains code and models used in the paper **"Analyzing the Evolution of Scientific Misconduct Based on the Language of Retracted Papers"** (ACL 2025).  
The goal is to detect and quantify scientific misconduct—especially from paper mills and AI-generated content—based on linguistic signals in retracted research papers.

The project introduces:
- A **mixture model framework** to estimate misconduct prevalence over time.
- Transformer-based classifiers that achieve high accuracy (up to **0.93 F1**) in detecting fraudulent writing.
- Tools to run large-scale inference on hundreds of thousands of abstracts to trace the **temporal trends** of scientific misconduct from 1980–2024.

---

## 📦 Installation

Set up a virtual environment and install required dependencies:

```bash
python3 -m venv mmenv
source mmenv/bin/activate
pip install -r requirements_mm.txt
```

If you want to re-run tokenization:

```bash
python -m spacy download en_core_web_lg
python mixture_model/tokenizer.py
```

---

## 🚀 Running Experiments

To train or modify the mixture model, edit the `DEFAULT_CONFIG` inside:

```python
mixture_model/train_new_distribution.py
```

Then run:

```bash
python mixture_model/train_new_distribution.py
```

This script estimates the prevalence of misconduct by fitting a distributional model that compares linguistic features between retracted and non-retracted papers.

---

## 🧠 Background

Scientific misconduct—particularly via **paper mills**—is a growing problem in academic publishing. These operations produce fraudulent manuscripts using ghostwriters, AI models, or auto-generators.

This project investigates whether **linguistic cues** can distinguish retracted papers (due to misconduct) from legitimate publications, using:

- Log-odds analysis of words and n-grams
- Type-token ratio to measure lexical diversity
- Mixture model estimation
- Transformer classifiers fine-tuned on domain-specific scientific corpora

---

## 📈 Key Results

- **Retracted papers** tend to use vague and repetitive language.
- Classifiers detect paper mill content with **>0.9 F1**.
- The model correlates strongly with real-world retraction trends (Pearson ρ = 0.79 for paper mills).
- Trend analysis reveals growing linguistic signals of misconduct across time and domains.

---

## 🔗 Important Links

- 📚 [Retraction Watch Blog](https://retractionwatch.com/)
- 📊 [Retraction Watch Dataset](https://gitlab.com/crossref/retraction-watch-data)
- 📂 [OpenAlex](https://openalex.org)

---

## 🧬 Related Work

- Liang et al. (2024) — [Monitoring AI-Modified Content at Scale](https://arxiv.org/pdf/2403.07183)
- Liang et al. (2024) — [Mapping LLM Use in Scientific Papers](https://arxiv.org/pdf/2404.01268)
- Razis et al. (2023) — Paper mill detection via transformer models

---

## ⚠️ Disclaimer

This tool is **not meant to accuse individual papers** of misconduct. The models are built for **collection-level analysis** and **trend estimation**, not forensic auditing.
