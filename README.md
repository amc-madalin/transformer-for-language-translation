# Transformer Model for Language Translation

## Overview
This project represents a personal journey in learning to code a Transformer model from scratch. The focus is on building a language translation system capable of translating between English and Italian. Using the Transformer architecture, known for its efficiency and effectiveness in natural language processing, this project aims to delve into the complexities of language translation in the machine learning field.

## Learning Objectives
- Understand and implement the Transformer model architecture.
- Explore the attention mechanism and its role in language translation.
- Gain hands-on experience with data preprocessing, model training, and evaluation in the context of NLP.

## Project Structure
- **Attention Visualization (`attention_visual.ipynb`)**: A Jupyter notebook used for visualizing the attention maps in the Transformer model, helping to understand how the model focuses on different parts of the sentence during translation.
- **Configuration Settings (`config.py`)**: Contains modifiable settings like hyperparameters, aiding in experimenting with different model configurations.
- **Dataset Processing (`dataset.py`)**: Manages loading and preprocessing of English and Italian language datasets for the translation task.
- **Model Architecture (`model.py`)**: Defines the Transformer model, constructed from scratch for the purpose of language translation.
- **Project Documentation (`README.md`)**: Provides an overview of the project, including setup and usage instructions.
- **Experiment Logs (`runs/`)**: Contains logs and outputs from various model training sessions.
- **Tokenizers (`tokenizer_en.json`, `tokenizer_it.json`)**: Tokenizers for English and Italian, transforming text into a machine-readable format.
- **Training Script (`train.py`)**: The main script for training the Transformer model, encapsulating the entire training loop.
- **Saved Model Weights (`weights/`)**: Stores the weights of the trained model for future use or reference.

## Installation
Detail the steps required to set up the project environment.

```bash
# Add installation commands here
