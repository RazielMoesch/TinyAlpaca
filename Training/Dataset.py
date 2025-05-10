import torch
from torch.utils.data import Dataset
from Tokenizer.tokenizer import BPETokenizer

class TinyPacaDataset(Dataset):
    def __init__(self, data_txt, tokenizer: BPETokenizer, seq_len=128, split="train", 
                 num_merges=100, vocab_save_path="vocab.json", load_vocab_path=None):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        
        if not load_vocab_path:
            self.tokenizer.train(data_txt, num_merges=num_merges, vocab_path=vocab_save_path)
        
        # Tokenize the entire text
        tokenized = self.tokenizer.tokenize(data_txt)
        tokens = [int(t) for t in (tokenized.split() if isinstance(tokenized, str) else tokenized)]
        
        # Split into non-overlapping chunks of seq_len
        self.examples = []
        for i in range(0, len(tokens) - seq_len, seq_len):
            chunk = tokens[i:i + seq_len]
            input_seq = torch.tensor(chunk[:-1], dtype=torch.long)
            target_seq = torch.tensor(chunk[1:], dtype=torch.long)
            self.examples.append((input_seq, target_seq))
        
        # Train/validation split
        split_idx = int(0.9 * len(self.examples))
        self.examples = self.examples[:split_idx] if split == "train" else self.examples[split_idx:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]