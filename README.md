# LLM Fine-Tuning, RAG, and Evaluation Framework

A  repository for implementing and evaluating state-of-the-art LLM techniques including fine-tuning, Retrieval-Augmented Generation (RAG), and model evaluation. This project demonstrates practical applications of modern NLP and LLM approaches across various use cases.

## Project Overview

This repository contains implementations of advanced LLM techniques focusing on:

- **Fine-tuning** open and closed source LLMs (BERT, Llama-3, FLAN-T5)
- **Retrieval-Augmented Generation (RAG)** systems with vector databases
- **Evaluation frameworks** for generative and understanding tasks
- **Knowledge distillation** and model optimization techniques
- **Prompt engineering** and LLM instruction alignment

## Repository Structure

### Directories
* `notebooks/`: Jupyter notebooks containing implementations and experiments
* `data/`: Datasets used in the notebooks

## Notebooks

### LLM Fine-Tuning

* `llm_fine_tuning.ipynb`: General approach to fine-tuning open and closed source LLMs
* `llama-3-8b-sft-rlf.ipynb`: Fine-tuning Llama-3 8B using domain-specific Q&A data
* `bert_fine_tuning_for_multilabel_classification.ipynb`: Fine-tuning BERT for anime category classification
* `09_flan_t5_rl.ipynb`: Reinforcement Learning techniques to improve FLAN-T5 model outputs

### Knowledge Distillation and Model Optimization

* `bert_distillation_example_1.ipynb`: Basic knowledge distillation methods for transformer models
* `bert_distillation_example_2.ipynb`: Advanced distillation techniques and applications

### Retrieval-Augmented Generation (RAG)

* `rag_chatbot.ipynb`: Implementation of a scalable RAG chatbot using Pinecone vector database
* `semantic_search_system.ipynb`: Building semantic search capabilities for information retrieval

### Recommendation Systems

* `recommendation_engine.ipynb`: Building recommendation systems using natural language descriptions

### Prompt Engineering

* `prompt_engineering_openai.ipynb`: Advanced LLM prompting techniques using OpenAI models

### Evaluation Frameworks

* `llm_generative_eval.ipynb`: Methods for evaluating the generative capabilities of LLMs

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.15+
- Datasets
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Pinecone
- OpenAI API (for certain notebooks)

### Installation

```bash
# Clone this repository
git clone https://github.com/yourusername/llm-fine-tuning-rag-eval.git

# Navigate to repository directory
cd llm-fine-tuning-rag-eval

# Create and activate virtual environment
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file in the root directory with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
PINECONE_API_KEY=your_pinecone_api_key
HUGGINGFACE_TOKEN=your_huggingface_token
```

## Usage Examples

### Fine-tuning Llama-3 with QLoRA

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3-8b", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8b")

# Configure LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

# Apply LoRA to model
model = get_peft_model(model, lora_config)

# See the notebook for full implementation details
```

### Building a RAG System with Pinecone

```python
import pinecone
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel

# Initialize Pinecone
pinecone.init(api_key="your_pinecone_api_key", environment="your_environment")
index = pinecone.Index("your_index_name")

# Create embeddings for your documents
# See rag_chatbot.ipynb for full implementation

# Query the system
query = "What is the capital of France?"
response = rag_system.query(query)
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## Acknowledgments

* Hugging Face for transformer models and libraries
* OpenAI for API access
* Meta for releasing Llama models
* Pinecone for vector database services
