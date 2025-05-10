import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Tokenizer import BPETokenizer

tokenizer = BPETokenizer()

example_txt = "Examples/Tokenizer/example.txt"
with open(example_txt, 'r', encoding='utf-8') as f:
    training_text = f.read()

vocab = tokenizer.train(training_text, num_merges=50, vocab_path="Examples/Tokenizer/example_vocab.json")

test_text = "Hello. This is an example text script. My name is Harry Potter."


token_ids = tokenizer.tokenize(test_text)
print(f"Tokens: {token_ids}")

detokenized = tokenizer.detokenize(token_ids)
print(f"Detokenized: {detokenized}")
