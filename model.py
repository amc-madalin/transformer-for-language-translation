import torch
import torch.nn as nn

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x: torch.Tensor):
        return self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
    
class PositionalEncoding(nn.Module):
    
    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        
        # Create a matrix of shape (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)
        # Create a vetor of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        # Apply sine to even indices in the range of d_model
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in the range of d_model
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension to the positional encoding
        self.pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(d_model), requires_grad=True)
        
    def forward(self, x: torch.Tensor):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.beta
    
class FeedForwardBlock(nn.Module):
        
        def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
            super().__init__()
            self.d_model = d_model
            self.d_ff = d_ff
            self.dropout = nn.Dropout(dropout)
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.relu = nn.ReLU()
            
        def forward(self, x: torch.Tensor):
            x = self.dropout(self.relu(self.linear1(x)))
            return self.linear2(x)
        
class MultiHeadAttentionBlock(nn.Module):
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.dropout = nn.Dropout(dropout)
        self.head_dim = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
   
    @staticmethod     
    def attention(q, k, v, mask, dropout: nn.Dropout):
        head_dim = q.shape[-1]
        
        # (batch_size, num_heads, seq_len, head_dim) * (batch_size, num_heads, head_dim, seq_len) = (batch_size, num_heads, seq_len, seq_len
        attention_scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(head_dim, dtype=torch.float32))
        
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
            
        attention_scores = torch.softmax(attention_scores, dim=-1) # (batch_size, num_heads, seq_len, seq_len)
        
        if dropout is not None:
            attention_scores = dropout(attention_scores)
            
        return torch.matmul(attention_scores, v), attention_scores
            
        
    def forward(self, q, k, v, mask):
        querry = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)
        
        # Split the querry, key and value into num_heads
        # Shape: (batch_size, seq_len, num_heads, head_dim)
        querry = querry.view(querry.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        key = key.view(key.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        value = value.view(value.shape[0], -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3) # (batch_size, num_heads, seq_len, head_dim)
        
        x, self.attention_scores = self.attention(querry, key, value, mask, self.dropout)
        
        # Concatenate the heads
        # Shape: (batch_size, seq_len, d_model)
        x = x.permute(0, 2, 1, 3).contiguous().view(x.shape[0], -1, self.d_model)
        
        return self.w_o(x)
    
class ResidualConnection(nn.Module):
    
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization(d_model)
        
    def froward(self, x: torch.Tensor, sublayer: nn.Module):
        return x + self.dropout(sublayer(self.layer_norm(x)))