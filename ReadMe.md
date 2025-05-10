TinyAlpaca: A Lightweight Transformer for Text Generation
TinyAlpaca is a compact transformer-based language model designed for text generation, trained on the Tiny Shakespeare dataset. This project implements a decoder-only transformer architecture, inspired by models like GPT, but scaled down for educational and experimental purposes. It includes a custom Byte Pair Encoding (BPE) tokenizer, dataset processing, training pipeline, and a text generation script.
Project Overview
TinyAlpaca is built to demonstrate the core components of a transformer model, including token and positional embeddings, masked multi-head self-attention, feed-forward networks, and layer normalization. The model is lightweight, with a small embedding dimension (d_model=64) and few layers (num_layers=4), making it suitable for quick training and experimentation on modest hardware.
Key Features
Note: This md was made with grokAI since its easier and better.

Decoder-Only Transformer: Autoregressive architecture for next-token prediction.
Custom BPE Tokenizer: Processes text into subword tokens, trained on the input dataset.
Tiny Shakespeare Dataset: ~1MB of Shakespearean text for training and validation.
Text Generation: Generates Shakespearean-style text from user-provided prompts.
PyTorch Implementation: Modular codebase with clear separation of model components.

Model Structure
TinyAlpaca follows a standard decoder-only transformer architecture, with the following components:
1. Embedding Layer (Embedding.py)

Purpose: Converts input token IDs into dense vectors and adds positional information.
Components:
Token Embeddings: Learnable embeddings for each token in the vocabulary (vocab_size x d_model).
Positional Embeddings: Learnable encodings for each position up to max_seq_len (max_seq_len x d_model).


Output: [batch_size, seq_len, d_model] tensor combining token and positional embeddings.

2. Masked Multi-Head Self-Attention (MaskedMultiHeadSelfAttention.py)

Purpose: Computes attention scores, allowing the model to focus on relevant tokens while respecting the autoregressive property (future tokens are masked).
Components:
QKV Projection: Linear layer projecting input to queries, keys, and values (d_model to 3 * d_model).
Multi-Head Attention: Splits d_model into num_heads heads, each with dimension d_model / num_heads.
Causal Mask: Triangular mask ensures attention only to previous tokens.
Output Projection: Linear layer to combine attention outputs (d_model to d_model).


Output: [batch_size, seq_len, d_model] tensor.

3. Feed-Forward Network (FeedForwardNetwork.py)

Purpose: Applies position-wise transformations to enhance model expressivity.
Components:
Linear Layers: Two layers (d_model to d_ff, d_ff to d_model).
ReLU Activation: Non-linearity between linear layers.


Output: [batch_size, seq_len, d_model] tensor.

4. Decoder Layer (DecoderLayer.py)

Purpose: Combines attention and feed-forward components with residual connections and normalization.
Components:
Masked Multi-Head Self-Attention: Computes attention over the input sequence.
Feed-Forward Network: Applies position-wise transformations.
Layer Normalization: Two normalization layers (LayerNorm) stabilize training.
Residual Connections: Add input to sub-layer outputs to prevent vanishing gradients.


Output: [batch_size, seq_len, d_model] tensor.

5. TinyAlpaca Model (Transformer.py)

Purpose: Orchestrates the full transformer pipeline.
Components:
Embedding Layer: Converts token IDs to embeddings with positional information.
Decoder Layers: Stack of num_layers decoder layers.
Final Linear Layer: Projects output to vocabulary size (d_model to vocab_size).


Input: Token IDs [batch_size, seq_len].
Output: Logits [batch_size, seq_len, vocab_size] for next-token prediction.

Hyperparameters

d_model = 64: Embedding and hidden dimension.
d_ff = 256: Feed-forward network dimension.
num_heads = 4: Number of attention heads.
num_layers = 4: Number of decoder layers.
max_seq_len = 128: Maximum sequence length.
vocab_size: Determined by tokenizer (e.g., 216 for Tiny Shakespeare).

Installation
Prerequisites

Python 3.8+
PyTorch 2.0+
Additional dependencies: tqdm (for progress bars), torchvision (optional).

Setup

Clone the repository:git clone https://github.com/yourusername/TinyAlpaca.git
cd TinyAlpaca


Install dependencies:pip install torch tqdm


Ensure the Tiny Shakespeare dataset is available at Examples/Model_Training/TinyShakespeare.txt. If not, download it or use another text corpus.

Usage
Directory Structure
TinyAlpaca/
├── Examples/
│   └── Model_Training/
│       ├── TinyShakespeare.txt      # Training dataset
│       ├── training_vocab.json      # Tokenizer vocabulary
│       ├── tinyalpaca.pth          # Trained model weights
│       ├── training_example.py      # Training script
│       └── test_tinyalpaca.py       # Testing/generation script
├── Model/
│   ├── Embedding.py                 # Embedding layer
│   ├── FeedForwardNetwork.py        # Feed-forward network
│   ├── MaskedMultiHeadSelfAttention.py # Attention mechanism
│   ├── DecoderLayer.py              # Decoder layer
│   └── Transformer.py               # TinyAlpaca model
├── Tokenizer/
│   └── tokenizer.py                 # BPE tokenizer
├── Training/
│   ├── Dataset.py                   # Dataset class
│   └── train.py                     # Training function
└── README.md

Training

Run the training script:python Examples/Model_Training/training_example.py


The script:
Trains the tokenizer on TinyShakespeare.txt if training_vocab.json doesn’t exist.
Creates training and validation datasets (90/10 split).
Trains the model for 10 epochs with AdamW optimizer (lr=3e-4).
Saves the model weights to tinyalpaca.pth.



Training Parameters (editable in training_example.py):

EPOCHS = 10
BATCH_SIZE = 32
D_MODEL = 64
D_FF = 256
NUM_HEADS = 4
NUM_LAYERS = 4
LEARNING_RATE = 3e-4
MAX_SEQ_LEN = 128

Example Output:
Vocab size: 216, Max token ID: 215
Epoch [1/10] | Train Loss: 4.0689 | Val Loss: 3.5343 | LR: 3.00e-04
...
Epoch [10/10] | Train Loss: 2.6376 | Val Loss: 2.7999 | LR: 3.00e-04

Testing/Generation

Run the generation script:python Examples/Model_Training/test_tinyalpaca.py


The script:
Loads the trained model (tinyalpaca.pth) and tokenizer (training_vocab.json).
Generates text from a prompt (default: "To be or not to be, that is the question:").
Outputs up to 100 new tokens with temperature sampling (temperature=0.7).



Example Output:
Generated Text:
To be or not to be, that is the question:
He worthy self presence take of the thatherse thy prince,
These begain that he be in the crow; for all a honour of his
So to speace. Who 't to should thee cant

Customization:

Edit test_tinyalpaca.py to change the prompt, max_new_tokens, or temperature.

Documentation
Key Files

Training/Dataset.py:

Implements TinyPacaDataset, which tokenizes the input text, splits it into sentences, and creates input-target pairs for training.
Splits data into 90% training and 10% validation.


Training/train.py:

Defines the train function, handling model training and validation loops.
Uses AdamW optimizer, cross-entropy loss, and gradient clipping (max_grad_norm=1.0).


Tokenizer/tokenizer.py:

Implements BPETokenizer, a custom BPE tokenizer that trains on the input text (100 merges by default).
Saves/loads vocabulary to/from training_vocab.json.


Model/ Directory:

Modular implementation of transformer components (see Model Structure above).


Examples/Model_Training/training_example.py:

Main training script, tying together dataset, tokenizer, model, and training logic.


Examples/Model_Training/test_tinyalpaca.py:

Generates text using the trained model, demonstrating its capabilities.



Extending the Project

Larger Model: Increase D_MODEL (e.g., 128 or 256) and NUM_LAYERS (e.g., 6 or 8) in training_example.py for better performance (requires more compute).
More Data: Replace TinyShakespeare.txt with a larger corpus (e.g., Project Gutenberg texts) to improve generalization.
Advanced Generation: Add top-k or top-p sampling in test_tinyalpaca.py for more controlled text generation.
Learning Rate Scheduling: Uncomment and configure the CosineAnnealingWarmRestarts scheduler in training_example.py for adaptive learning rates.

Performance Notes

Training Time: ~35 seconds for 10 epochs on a GPU, due to small model size and dataset.
Loss: Train loss drops from ~4.07 to ~2.64; validation loss from ~3.53 to ~2.80.
Generated Text: Semi-coherent, Shakespearean-style output, but limited by model capacity and dataset size.

Limitations

Model Capacity: Small dimensions (d_model=64, num_layers=4) limit the model’s ability to capture complex linguistic patterns.
Dataset Size: Tiny Shakespeare (~1MB) is small, leading to overfitting and limited vocabulary.
Generation Quality: Output may include non-words or lack full coherence due to limited training.

Contributing
Contributions are welcome! To contribute:

Fork the repository.
Create a feature branch (git checkout -b feature-name).
Commit changes (git commit -m "Add feature").
Push to the branch (git push origin feature-name).
Open a pull request.

License
This project is licensed under the MIT License. See LICENSE for details.
Acknowledgments

Inspired by transformer architectures like GPT and BERT.
Tiny Shakespeare dataset provided by Andrej Karpathy’s nanoGPT.
Built with PyTorch and love for natural language processing.


Happy Shakespearean text generation!
