import torch
import torch.nn as nn
from model import Transformer
from dataset import get_dataloader, make_src_mask, make_tgt_mask
from config import Config

class TransformerLRScheduler:
    """Implements the learning rate schedule from Section 5.3 of the paper:
    lr = d_model^{-0.5} * min(step^{-0.5}, step * warmup_steps^{-1.5})
    """
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self._step = 0

    def step(self):
        self._step += 1
        lr = self._get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self):
        return self.d_model ** (-0.5) * min(
            self._step ** (-0.5),
            self._step * self.warmup_steps ** (-1.5)
        )

def train():
    print(f"Using device: {Config.device}")
    
    dataloader = get_dataloader(Config.batch_size, num_samples=3000, seq_len=15, vocab_size=Config.src_vocab_size)
    
    model = Transformer(
        src_vocab_size=Config.src_vocab_size,
        tgt_vocab_size=Config.tgt_vocab_size,
        d_model=Config.d_model,
        num_heads=Config.num_heads,
        num_layers=Config.num_layers,
        d_ff=Config.d_ff,
        max_seq_len=Config.max_seq_len,
        dropout=Config.dropout
    ).to(Config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = TransformerLRScheduler(optimizer, Config.d_model, warmup_steps=Config.warmup_steps)
    criterion = nn.CrossEntropyLoss(ignore_index=Config.pad_idx)
    
    model.train()
    for epoch in range(Config.epochs):
        epoch_loss = 0
        
        for batch_idx, (src, tgt) in enumerate(dataloader):
            src = src.to(Config.device)
            tgt = tgt.to(Config.device)
            
            tgt_input = tgt[:, :-1]
            tgt_expected = tgt[:, 1:]
            
            src_mask = make_src_mask(src, Config.pad_idx).to(Config.device)
            tgt_mask = make_tgt_mask(tgt_input, Config.pad_idx).to(Config.device)
            
            optimizer.zero_grad()
            output = model(src, tgt_input, src_mask, tgt_mask)
            
            output = output.reshape(-1, output.shape[-1])
            tgt_expected = tgt_expected.reshape(-1)
            
            loss = criterion(output, tgt_expected)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        current_lr = scheduler._get_lr()
        print(f"Epoch {epoch+1}/{Config.epochs} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f}")
    
    # Quick evaluation on one batch
    model.eval()
    with torch.no_grad():
        src, tgt = next(iter(dataloader))
        src, tgt = src.to(Config.device), tgt.to(Config.device)
        tgt_input, tgt_expected = tgt[:, :-1], tgt[:, 1:]
        src_mask = make_src_mask(src, Config.pad_idx).to(Config.device)
        tgt_mask = make_tgt_mask(tgt_input, Config.pad_idx).to(Config.device)
        output = model(src, tgt_input, src_mask, tgt_mask)
        preds = output.argmax(dim=-1)
        accuracy = (preds == tgt_expected).float().mean().item()
        print(f"\nCopy-task accuracy on one batch: {accuracy:.2%}")

if __name__ == "__main__":
    train()
