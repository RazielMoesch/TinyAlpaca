import torch
from torch.utils.data import Dataset
from Tokenizer.tokenizer import BPETokenizer


class TinyPacaDataset(Dataset):
    def __init__(self, data_txt, tokenizer: BPETokenizer, seq_len=128, split="train", num_merges=100, vocab_save_path="vocab.json", load_vocab_path=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        if not load_vocab_path:
            self.tokenizer.train(data_txt, num_merges=num_merges, vocab_path=vocab_save_path)
        
        tokenized = self.tokenizer.tokenize(data_txt)
        self.tokens = [int(t) for t in (tokenized.split() if isinstance(tokenized, str) else tokenized)]
        
        # More thorough validation
        vocab_size = len(self.tokenizer.vocab)
        bad_tokens = [t for t in self.tokens if t >= vocab_size]
        if bad_tokens:
            print(f"Found {len(bad_tokens)} invalid tokens (max allowed: {vocab_size-1})")
            print(f"Example bad tokens: {bad_tokens[:10]}")
            # Handle bad tokens by clipping them
            self.tokens = [min(t, vocab_size-1) for t in self.tokens]
        
        split_idx = int(0.9 * len(self.tokens))
        self.tokens = self.tokens[:split_idx] if split == "train" else self.tokens[split_idx:]
        
        self.examples = []
        for i in range(0, len(self.tokens) - self.seq_len - 1):
            input_seq = torch.tensor(self.tokens[i:i+self.seq_len], dtype=torch.long)
            target_seq = torch.tensor(self.tokens[i+1:i+1+self.seq_len], dtype=torch.long)
            self.examples.append((input_seq, target_seq))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]