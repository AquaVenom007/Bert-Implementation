# BERT From Scratch and Fakeddit Classification

A Natural Language Processing project implementing the core components of a BERT-style encoder from scratch using PyTorch, followed by preprocessing of the Fakeddit dataset for fake-news classification.

## Overview

This project is based on an NLP assignment with three main tasks:

1. Implement a BERT-style encoder from scratch.
2. Preprocess the Fakeddit dataset for fake-news classification.
3. Fine-tune `bert-base-uncased` for binary classification and report evaluation metrics.

The notebook currently contains the implementation and testing for **Task 1** and the preprocessing pipeline for **Task 2 (Fakeddit)**.

## Task 1: BERT Implementation from Scratch

The BERT model is implemented using PyTorch without directly using a pretrained BERT model for the encoder architecture.

### Model Configuration

| Parameter | Value |
|---|---:|
| Embedding Dimension | 768 |
| Number of Attention Heads | 12 |
| Number of Encoder Layers | 2 |
| Feed-Forward Dimension | 3072 |
| Vocabulary Size | 30,522 |
| Maximum Sequence Length | 512 |
| Segment Types | 2 |
| Number of Classes | 2 |

### Components Implemented

The custom BERT model contains:

- Multi-Head Self-Attention
- Feed-Forward Network
- Transformer Encoder Layer
- Token Embeddings
- Positional Embeddings
- Segment Embeddings
- Layer Normalization
- Dropout
- Pooler
- Binary Classification Head

### Multi-Head Attention

The attention module performs:

1. Query, Key, and Value projections
2. Splitting of embeddings into 12 attention heads
3. Scaled dot-product attention
4. Attention masking
5. Softmax normalization
6. Context-vector computation
7. Concatenation of attention heads
8. Final linear projection

### Transformer Encoder Layer

Each encoder layer consists of:

- Multi-Head Self-Attention
- Residual connection and Layer Normalization
- Feed-Forward Network with GELU activation
- Residual connection and Layer Normalization

Two encoder layers are stacked in the custom BERT model.

### BERT Embeddings

The input representation is constructed by combining:

- Token embeddings
- Positional embeddings
- Segment embeddings

The combined representation is normalized and passed through dropout.

### Pooler and Classifier

The hidden representation corresponding to the first token (`[CLS]`) is passed through:

- A linear layer
- Tanh activation
- A final linear classification layer

The classifier produces two output logits for binary classification.

## Tokenizer

The project uses the Hugging Face:

`bert-base-uncased`

tokenizer for converting the sample sentence into BERT token IDs.

A sample sentence containing more than 10 words is tokenized and padded or truncated to a maximum length of 64 tokens for testing.

Segment IDs are set to zero for all tokens as specified for the sample test.

## Task 1: Model Testing

The implemented model was tested using a sample sentence.

### Reported Tensor Shapes

| Stage | Output Shape |
|---|---|
| Token Embeddings | `(1, 64, 768)` |
| Multi-Head Attention | `(1, 64, 768)` |
| Feed-Forward Layer | `(1, 64, 768)` |
| Final Encoder Output | `(1, 64, 768)` |
| Classifier Output | `(1, 2)` |

The attention and feed-forward outputs are reported for both encoder layers.

### Trainable Parameters

The custom two-layer BERT-style model contains approximately:

**38.61 million trainable parameters**

This is smaller than the approximately 110 million parameters of the full BERT-base model because this implementation uses only two encoder layers.

## Task 2: Fakeddit Data Preprocessing

The notebook also contains a preprocessing pipeline for the Fakeddit dataset for binary fake-news classification.

### Dataset Columns

The preprocessing pipeline uses:

- `title`
- `clean_comment`
- `2_way_label`

The label is interpreted as:

- `0` = Real
- `1` = Fake

### Preprocessing Steps

#### 1. Load Dataset

The Fakeddit data is loaded as a tab-separated file using Pandas.

#### 2. Combine Text

The `title` and `clean_comment` fields are combined into a single `text` column.

Missing comments are handled using empty strings.

#### 3. Remove Duplicates

As a bonus preprocessing step, duplicate posts with identical combined text are removed.

#### 4. Balance the Dataset

An equal number of real and fake posts are randomly sampled.

The target is at least:

- 2,500 real posts
- 2,500 fake posts

Sampling uses a fixed random seed of `42`.

#### 5. Train-Test Split

The balanced dataset is divided into:

- 80% training
- 20% testing

Stratified sampling is used to preserve the class balance.

### Reproducibility

The preprocessing pipeline uses:

- Random seed: `42`
- Test size: `20%`
- Minimum samples per class: `2,500`

## Task 3: Fine-Tuning

The assignment specifies fine-tuning `bert-base-uncased` for binary classification using cross-entropy loss.

The required evaluation metrics are:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

The current notebook does not contain the Task 3 fine-tuning implementation yet.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Pandas
- Scikit-learn
- Google Colab

## Project Structure

bert-implementation/
├── bert_implementation.ipynb
└── README.md

## How to Run

1. Open `bert_implementation.ipynb` in Google Colab or Jupyter Notebook.
2. Install the required packages.
3. Run the import and model implementation cells.
4. Run the Task 1 testing cell to initialize the tokenizer and custom BERT model.
5. For Task 2, update the `FILE_PATH` variable to point to your local Fakeddit `all_comments.tsv` file.
6. Run the preprocessing pipeline.
7. Implement and run the Task 3 fine-tuning section if required.

## Important Note

The Fakeddit preprocessing code currently contains a local Windows file path:

`D:\NLP\assignment 3\all_comments.tsv`

This path must be changed to the location of `all_comments.tsv` on the system where the notebook is executed.

## Future Improvements

- Complete the WildChat preprocessing pipeline
- Complete the Fakeddit post-comment-reply structure
- Implement BERT fine-tuning for classification
- Add training and validation curves
- Report accuracy, precision, recall, and F1-score
- Generate and visualize the confusion matrix
- Compare the custom BERT encoder with pretrained `bert-base-uncased`

## Acknowledgements

This project uses the `bert-base-uncased` tokenizer from Hugging Face Transformers and follows the requirements of the NLP assignment for implementing BERT components and preparing text classification datasets.
