import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from Tokenizer.tokenizer import BPETokenizer
from Model.Transformer import TinyAlpaca
from tqdm import tqdm



def train(epochs, train_dl, val_dl, d_model, d_ff, num_heads, vocab_size,
          num_layers, optimizer, loss_fn, lr_scheduler, device):
    model = TinyAlpaca(d_model, d_ff, num_heads, vocab_size, num_layers).to(device)
    model.train()

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0.0

        for  (input_ids, target_ids) in tqdm((train_dl), desc="Epoch Progress:", leave=False):
            input_ids = input_ids.to(device).squeeze(1)  # Remove extra dimension if needed
            target_ids = target_ids.to(device).squeeze(1)
            
            optimizer.zero_grad()
            output = model(input_ids)
            loss = loss_fn(output.view(-1, vocab_size), target_ids.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for(input_ids, target_ids) in (val_dl):
                input_ids = input_ids.to(device).squeeze(1)
                target_ids = target_ids.to(device).squeeze(1)
                output = model(input_ids)
                loss = loss_fn(output.view(-1, vocab_size), target_ids.view(-1))
                val_loss += loss.item()

        avg_train_loss = epoch_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        print(f'Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        if lr_scheduler:
            lr_scheduler.step()