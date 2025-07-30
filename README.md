# Analyzing the Evolution of Scientific Misconduct Based on the Language of Retracted Papers
Analyzing the Evolution of Scientific Misconduct based on the Language of Retracted Papers


## Mixture Model Max Likelihood Estimation and Inference

### Install
```
python3 -m venv mmenv
source mmenv/bin/activate
pip install -r requirements_mm.txt
```

If you want to rerun tokenization:
```
python -m spacy download en_core_web_lg
python mixture_model/tokenizer.py
```

### Run Experiments
In `mixture_model/train_new_distribution.py` copy DEFAULT_CONFIG and change values.
```
python mixture_model/train_new_distribution.py
``` 

## Important Links
- [Retraction Watch Blog](https://retractionwatch.com/)
- [Retraction Watch Dataset](https://gitlab.com/crossref/retraction-watch-data)

## Related Work
- Weixin Liang et al.: [Monitoring AI-Modified Content at Scale:
A Case Study on the Impact of ChatGPT on AI Conference Peer Reviews](https://arxiv.org/pdf/2403.07183)
- Weixin Liang et al. [Mapping the Increasing Use of LLMs in Scientific Papers](https://arxiv.org/pdf/2404.01268)