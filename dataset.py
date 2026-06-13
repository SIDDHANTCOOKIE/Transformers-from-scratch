import torch
from torch.utils.data import Dataset, DataLoader

def make_src_mask(src, pad_idx):
    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)
    return src_mask

def make_tgt_mask(tgt, pad_idx):
    tgt_pad_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    tgt_len = tgt.shape[1]
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()
    tgt_mask = tgt_pad_mask & tgt_sub_mask
    return tgt_mask

class SyntheticDataset(Dataset):
    def __init__(self, num_samples, seq_len, vocab_size):
        self.data = [torch.randint(1, vocab_size, (seq_len,)) for _ in range(num_samples)]
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        return self.data[idx], self.data[idx].clone()

def get_dataloader(batch_size, num_samples=1000, seq_len=10, vocab_size=50):
    dataset = SyntheticDataset(num_samples, seq_len, vocab_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
