
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

## Usage
To effectively use this project, follow these steps:

1. **Clone the Repository**:
   Begin by cloning the repository to your local machine. Use the following command, replacing `<repository_url>` with the actual URL of the repository:

   ```bash
   git clone <repository_url>
   ```

2. **Create a Python Environment**:
   It's recommended to use a Conda environment for this project. Create a new environment named `transformer` with the following command:

   ```bash
   conda create --name transformer python=3.x
   ```

   Replace `3.x` with the version of Python you wish to use.

3. **Activate the Environment**:
   Once the environment is set up, activate it using:

   ```bash
   conda activate transformer
   ```

4. **Install Dependencies**:
   The project's dependencies are listed in the `requirements.txt` file. Install them by running:

   ```bash
   pip install -r requirements.txt
   ```

5. **Prepare the Data**:
   The dataset required for training is automatically downloaded by the project. The source (`lang_src`) and target (`lang_tgt`) languages can be configured in the `config.py` script. By default, it is set for English to Italian translation:

   ```python
   "lang_src": "en",
   "lang_tgt": "it",
   ```

6. **Train the Model**:
   Start the training process with the `train.py` script:

   ```bash
   python train.py
   ```

7. **Use the Model**:
   After training, the model's weights are saved in the `weights/` directory. These weights can be used for making predictions, evaluations, and further applications of the model.

## Learning Resources

Youtube Link - [Coding a Transformer from scratch on PyTorch, with full explanation, training and inference.](https://www.youtube.com/watch?v=ISNdQcPhsts&t=6932s)

## Acknowledgements
Credits to any individual or resource that has significantly aided in this learning project.

- **Umar Jamil**
