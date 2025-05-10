import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from Tokenizer.tokenizer import BPETokenizer
from Model.Transformer import TinyAlpaca
from tqdm import tqdm


def train(epochs, model, train_dl, val_dl, d_model, d_ff, num_heads, vocab_size,
          num_layers, optimizer, loss_fn, lr_scheduler, device):
    # Initialize model with proper device placement
    
    
    # Gradient clipping to prevent explosions
    max_grad_norm = 1.0
    
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        epoch_loss = 0.0
        
        for input_ids, target_ids in tqdm(train_dl, desc="Training", leave=False):
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            
            # Verify input shapes
            assert input_ids.dim() == 2, f"Expected [batch, seq_len], got {input_ids.shape}"
            assert target_ids.dim() == 2, f"Expected [batch, seq_len], got {target_ids.shape}"
            
            
            output = model(input_ids)
            
            # Verify output shape
            assert output.shape[:2] == input_ids.shape, \
                f"Output shape {output.shape} doesn't match input {input_ids.shape}"
            
            loss = loss_fn(output.view(-1, vocab_size), target_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            optimizer.step()
            epoch_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for input_ids, target_ids in tqdm(val_dl, desc="Validation", leave=False):
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)
                
                output = model(input_ids)
                loss = loss_fn(output.view(-1, vocab_size), target_ids.view(-1))
                val_loss += loss.item()

        # Calculate metrics
        avg_train_loss = epoch_loss / len(train_dl)
        avg_val_loss = val_loss / len(val_dl)
        
        # Learning rate scheduling
        if lr_scheduler:
            lr_scheduler.step(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{epochs}] '
              f'| Train Loss: {avg_train_loss:.4f} '
              f'| Val Loss: {avg_val_loss:.4f} '
              f'| LR: {optimizer.param_groups[0]["lr"]:.2e}')
        
        # Early stopping check
        if avg_val_loss > 10:  # Unreasonably high loss
            print("Abnormal training detected - stopping early")
            break

    return model