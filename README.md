<h1 align="center">mBERT vs MuRIL: Hinglish Hate Speech</h1>

**Run a matched binary Hinglish hate-speech baseline and inspect the training and evaluation choices behind it.**

Python training scripts. External CSV and model downloads required. Single split; evaluation also guides checkpoint selection.

By Om Patnaik · repository created December 2025.

## Install

| Platform / path | Start here |
| --- | --- |
| Python | Create a virtual environment; the training runner installs its Python dependencies |

Provide a CSV with `text` and `hate_label`, then use the training command below. To inspect the method without running training, read [the expanded study](https://github.com/OmTheLast/mBERT-vs-MuRIL-cross-dataset-hinglish-hate).

```bash
python -m compileall -q Training benchmark.py harness.py
```

## Why

You want to compare multilingual pretraining with Indian-language pretraining under the same initial recipe. This baseline fine-tunes mBERT and MuRIL with an 80/20 seed-42 split, two epochs, batch size 16, a 2e-5 learning rate, and 128-token inputs. It does not establish cross-domain generalization.

## Prepare the baseline input

Cleaning lowercases text, removes URLs, handles and punctuation, and normalizes whitespace. The training path does not automatically filter languages; inspect your CSV before using it.

## Inspect the saved model

Outputs are `Hinglish_Hate_Model_mBert` and `Hinglish_Hate_Model_MuRIL`. Update `benchmark.py` paths before benchmarking; its defaults point elsewhere. The 26 released checkpoints belong to the later cross-dataset study.

## Notes from building it

I began with one matched recipe so the pretraining choice was explicit. The later study added datasets, seeds, mixtures, and error analysis after this single-split comparison proved too narrow. I would treat this repo as the baseline method rather than pool it with independent evidence.

## Commands

```bash
python Training/Trainer.py --model both --data-path combined_hate_speech_dataset.csv  # train both models
python benchmark.py                         # benchmark after updating model paths
python -m compileall -q Training benchmark.py harness.py  # check syntax without training
```

## Verify it yourself

```bash
python -m compileall -q Training benchmark.py harness.py
```

Compilation checks syntax and launches no training. Benchmarking requires local checkpoints and its diagnostic inputs; the small diagnostic set is not an independent moderation benchmark.

## Credits, license and pointers

Code and analysis by Om Patnaik, assisted by AI tools. Builds on Google mBERT/MuRIL, Transformers, and PyTorch. No source license file is present; checkpoint terms are documented in the later study’s model cards.

[Full setup, original notes, and project history](README-DETAILS.md).
