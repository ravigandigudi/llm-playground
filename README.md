# 🤖 LLM Playground

<div align="center">
  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗_Hugging_Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)

</div>

An experimental platform for fine-tuning, evaluating, and deploying custom Large Language Models (LLMs). This repository serves as a playground for exploring various LLM architectures, fine-tuning techniques, and deployment strategies.

## 🌟 Features

- **Model Training & Fine-tuning**: Scripts for fine-tuning pre-trained models on custom datasets
- **Evaluation Framework**: Comprehensive evaluation tools to assess model performance
- **Inference API**: FastAPI-based API for model inference
- **Prompt Engineering**: Tools and examples for effective prompt engineering
- **RAG Integration**: Retrieval-Augmented Generation implementation 
- **Deployment Utilities**: Tools for optimizing and deploying models

## 🏗️ Project Structure

```
llm-playground/
├── data/                      # Sample and processed data 
├── models/                    # Model definitions and configurations
├── notebooks/                 # Jupyter notebooks for experimentation
├── src/                       # Source code
│   ├── evaluation/            # Evaluation metrics and tools
│   ├── fine_tuning/           # Fine-tuning scripts
│   ├── inference/             # Inference API
│   ├── rag/                   # Retrieval-Augmented Generation
│   └── utils/                 # Utility functions
├── tests/                     # Unit tests
├── configs/                   # Configuration files
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- PyTorch 2.0+
- CUDA-compatible GPU (recommended for training)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/Saavnbeli/llm-playground.git
   cd llm-playground
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up the configuration:
   ```bash
   cp configs/config.example.yaml configs/config.yaml
   # Edit config.yaml with your settings
   ```

## 💡 Usage Examples

### Fine-tuning a Model

```python
from src.fine_tuning.trainer import Trainer
from src.utils.config import load_config

config = load_config("configs/finetune_config.yaml")
trainer = Trainer(config)
trainer.train()
```

### Running Inference

```python
from src.inference.model import LLMModel

model = LLMModel("models/finetuned-model")
response = model.generate("What is the capital of France?")
print(response)
```

### Starting the API Server

```bash
uvicorn src.inference.api:app --host 0.0.0.0 --port 8000
```

## 📈 Benchmarks

| Model | Dataset | Accuracy | F1 Score | Latency (ms) |
|-------|---------|----------|----------|--------------|
| Mistral-7B | GSM8K | 78.3% | 0.81 | 145 |
| Llama2-7B | GSM8K | 73.1% | 0.77 | 152 |
| Finetuned-Mistral-7B | GSM8K | 84.5% | 0.86 | 146 |

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- Hugging Face for their transformers library
- PyTorch team for the amazing framework
- The open-source LLM community

---

<div align="center">
  
Created with ❤️ by [Sawan Beli](https://github.com/Saavnbeli)

</div>

