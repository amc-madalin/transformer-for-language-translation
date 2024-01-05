import torch    
import torch.nn as nn
from torch.utils.data import Dataset
from datasets import load_dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len):
        super().__init__()
        
        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.Tensor(tokenizer_tgt.token_to_id("[SOS]"), dtype=torch.int64)
        self.eos_token = torch.Tensor(tokenizer_tgt.token_to_id("[EOS]"), dtype=torch.int64)
        self.pad_token = torch.Tensor(tokenizer_tgt.token_to_id("[PAD]"), dtype=torch.int64)
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        src_target_pair = self.ds[idx]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['translation'][self.tgt_lang]
        
        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids
        
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1
        
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise Exception("Sequence length is too short")
        
        encoder_input = torch.cat([
            self.sos_token,
            torch.Tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.Tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
        ])
        
        decoder_input = torch.cat([
            self.sos_token,
            torch.Tensor(dec_input_tokens, dtype=torch.int64),
            torch.Tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])
        
        label = torch.cat([
            torch.Tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.Tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
        ])
        
        assert encoder_input.shape == decoder_input.shape == label.shape == (self.seq_len,) 
        
        return {
            'encoder_input': encoder_input,
            'decoder_input': decoder_input,
            'encoder_mask': (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int(),
            'decoder_mask': (decoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int() & causal_mask(decoder_input.shape[0]), # (1, 1, seq_len, seq_len),
            'label': label,
            'src_text': src_text,
            'tgt_text': tgt_text
        }
        
def causal_mask(seq_len):
    mask = torch.triu(torch.ones(1, seq_len, seq_len), diagonal=1).type(torch.int)
    return mask == 0