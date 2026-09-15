# mBERT vs MuRIL: Hinglish Hate-Speech Detection

An initial comparison of mBERT and MuRIL for binary hate-speech detection in Hinglish (Hindi–English code-mixed text).

[Expanded cross-dataset study](https://github.com/OmTheLast/mBERT-vs-MuRIL-cross-dataset-hinglish-hate) · [Models on Hugging Face](https://huggingface.co/collections/OmTheLast/hinglish-research-mbert-vs-muril-6aa96a16ed03bba0e89e9408)

## Research question

How do multilingual pretraining (mBERT) and Indian-language pretraining (MuRIL) compare when fine-tuned with the same binary classification setup?

## Method

- Fine-tune `bert-base-multilingual-cased` and `google/muril-base-cased` on CSV data with `text` and `hate_label` columns.
- Lowercase text, remove URLs, handles and punctuation, and normalize whitespace.
- Use an 80/20 random split with seed 42, two epochs, batch size 16, learning rate 2e-5, and a 128-token limit.
- Record accuracy and weighted F1 during training; inspect precision, recall, F1 and inference latency with the benchmark script.

## Run training

```bash
git clone https://github.com/OmTheLast/mBERT-vs-MuRIL-in-detecting-hatespeech.git
cd mBERT-vs-MuRIL-in-detecting-hatespeech
python -m venv .venv
source .venv/bin/activate
python Training/Trainer.py --model both --data-path combined_hate_speech_dataset.csv
```

The runner installs its Python dependencies. Saved models are written to `Hinglish_Hate_Model_mBert` and `Hinglish_Hate_Model_MuRIL`.

For benchmarking, update the model paths in [benchmark.py](benchmark.py) to these output directories; its defaults point to `models/mbert_model` and `models/muril_model`.

## Scope and limitations

This is a single-split baseline. The evaluation split also guides checkpoint selection, so it is not an untouched final test set. The supplied training path does not filter languages automatically. The small diagnostic benchmark does not establish generalization or suitability for automated moderation.

The [later study](https://github.com/OmTheLast/mBERT-vs-MuRIL-cross-dataset-hinglish-hate) adds three datasets, multiple seeds, training mixtures, cross-dataset evaluation and error analysis. The **26 checkpoints in the linked Hugging Face collection belong to that later study**; its model cards document results, label definitions, loading instructions and licensing notes.

Code and analysis by **Om Patnaik**.

### Tools Note

AI tools were used for coding, debugging, and documentation assistance; the research direction, result interpretation, and final claims were reviewed and owned by Om Patnaik.
