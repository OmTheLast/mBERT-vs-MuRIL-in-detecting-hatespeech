# Hinglish Hate Speech Detection

This repository contains research and implementation for detecting hate speech in Hinglish (Hindi-English mixed language) text using transformer-based models. The project compares the performance of mBERT (Multilingual BERT) and MuRIL (Multilingual Representations for Indian Languages) models for hate speech classification.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

## Overview

Hinglish, a blend of Hindi and English, is widely used in social media and online platforms in India. Detecting hate speech in this mixed language presents unique challenges due to code-switching, transliteration, and cultural context. This project aims to evaluate and compare the effectiveness of multilingual transformer models in identifying hate speech in Hinglish text.

## Dataset

The dataset used in this project is a combined Hinglish hate speech dataset containing:
- Text content in Hinglish (Romanized script)
- Binary labels: 0 (Non-Hate) and 1 (Hate Speech)
- Various contexts including social media posts, comments, and other user-generated content

The dataset undergoes preprocessing including:
- URL removal
- User mention removal
- Special character cleaning
- Text normalization

### Dataset Challenges
We've identified some challenges with the current dataset that may impact model performance:
- Potential class imbalance between hate and non-hate samples
- Limited context for ambiguous expressions
- Possible annotation inconsistencies in mixed-language content
- Coverage gaps for certain dialects or expressions

### Future Work: Multi-Dataset Comparison
To address these challenges and provide more robust results, we plan to train and evaluate models on multiple Hinglish hate speech datasets. This will allow us to:
- Compare model performance across different data distributions
- Identify which models generalize better across various contexts
- Determine the impact of dataset quality and size on performance
- Establish more reliable benchmarks for Hinglish hate speech detection

## Models

### mBERT (Multilingual BERT)
- Pre-trained on 104 languages including Hindi
- 12-layer, 768-hidden, 12-heads, 110M parameters
- General-purpose multilingual model

### MuRIL (Multilingual Representations for Indian Languages)
- Specifically designed for Indian languages
- Enhanced performance on Indian language tasks
- Better representation of code-switched content

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hinglish-hate-speech-detection.git
cd hinglish-hate-speech-detection
```

2. Install required packages:
```bash
pip install transformers accelerate scikit-learn datasets pandas torch
```

Or run the package installation script:
```bash
python Training/Packages.py
```

## Usage

### Training Models

To train both models:
```bash
python Training/Trainer.py --model both
```

To train only MuRIL:
```bash
python Training/Trainer.py --model muril
```

To train only mBERT:
```bash
python Training/Trainer.py --model mbert
```

To specify a custom dataset path:
```bash
python Training/Trainer.py --model both --data-path /path/to/your/dataset.csv
```

### Training Individual Models

To train MuRIL separately:
```bash
python Training/Train_MuRIL.py
```

To train mBERT separately:
```bash
python Training/Train_mBert.py
```

### Loading Pre-trained Models

After training, the models are saved in:
- `./Hinglish_Hate_Model_MuRIL/` for MuRIL
- `./Hinglish_Hate_Model_mBert/` for mBERT

You can load them using the transformers library:
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# For MuRIL
tokenizer = AutoTokenizer.from_pretrained("./Hinglish_Hate_Model_MuRIL")
model = AutoModelForSequenceClassification.from_pretrained("./Hinglish_Hate_Model_MuRIL")

# For mBERT
tokenizer = AutoTokenizer.from_pretrained("./Hinglish_Hate_Model_mBert")
model = AutoModelForSequenceClassification.from_pretrained("./Hinglish_Hate_Model_mBert")
```

## Training

The training process includes:
- Data preprocessing and cleaning
- Train/test split (80/20)
- Tokenization with max length of 128
- Training with batch size of 16
- 2 epochs with learning rate of 2e-5
- Evaluation metrics: Accuracy and F1-score
- Model saving after training

### Hyperparameters
- Learning Rate: 2e-5
- Batch Size: 16
- Epochs: 2
- Max Sequence Length: 128
- Weight Decay: 0.01

## Results

The models are evaluated using:
- Accuracy
- F1-score (weighted average)
- Precision and Recall (available in detailed logs)

Detailed benchmark results are saved in `benchmark_results_detailed.csv`.

## Project Structure

```
Hinglish Research/
├── Training/
│   ├── Packages.py           # Package installation script
│   ├── Loading_Dataset.py    # Data loading and preprocessing
│   ├── Trainer.py           # Main training script (runs both models)
│   ├── Train_MuRIL.py       # MuRIL-specific training
│   └── Train_mBert.py       # mBERT-specific training
├── combined_hate_speech_dataset.csv  # Main dataset
├── benchmark_results_detailed.csv    # Detailed results
└── README.md                # This file
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to the creators of mBERT and MuRIL models
- Dataset contributors for providing Hinglish hate speech data
- Hugging Face for providing the transformers library
