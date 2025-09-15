# PcodeBERT: BERT-based Embedding for GNNs

This project utilizes a **BERT** model to generate embeddings for program code instructions. These embeddings are then fed into a series of **Graph Neural Networks (GNNs)** to perform downstream tasks.

## Getting Started

### Prerequisites

- Python 3.12
- You can install the required packages using pip:
`
pip install -r requirements.txt
`

### Usage

1. **Prepare your data**: Place your raw data in the `data/raw/` directory.
2. **Train the model**:
python src/pipelines/train.py

3. **Evaluate the model**:
python src/pipelines/evaluate.py


## Project Structure

- `src/`: Contains all the source code for data processing, models, and training pipelines.
- `data/`: Stores the raw and processed datasets.
- `checkpoints/`: Saved model weights and checkpoints.
- `results/`: Output files from model evaluation and predictions.
- `config/`: Configuration files for different experiments.

