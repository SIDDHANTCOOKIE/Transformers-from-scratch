import torch

class Config:
    # Model Architecture
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    dropout = 0.1
    max_seq_len = 100
    
    # Vocabulary sizes (Set to 50 for the synthetic task)
    src_vocab_size = 50 
    tgt_vocab_size = 50
    pad_idx = 0
    
    # Training Parameters
    batch_size = 64
    epochs = 20
    warmup_steps = 400
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
