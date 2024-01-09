import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import BilingualDataset, causal_mask
from model import build_transformer
from config import get_weights_file_path, get_config

from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

from pathlib import Path
from tqdm import tqdm

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_src.token_to_id("[SOS]")
    eos_idx = tokenizer_src.token_to_id("[EOS]")
    
    # Precompute the encoder output and reuse it for every token we get from the decoder
    encoder_output = model.encode(source, source_mask) # (batch_size, seq_len, d_model)
    
    # Initialize the decoder input with the SOS token
    decoder_input = torch.empty(1,1).fill_(sos_idx).type_as(source).to(device) # (batch_size, seq_len)
    while True:
        if decoder_input.size(1) == max_len:
            break
        
        # Build mask for target 
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device) # (batch_size, seq_len, seq_len)

        out = model.decode(decoder_input, encoder_output, source_mask, decoder_mask) # (batch_size, seq_len, d_model)
        
        # Get the next token
        prob = model.project(out[:,-1])
        
        # Select the token with max probability
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat([decoder_input, torch.empty(1,1).type_as(source).fill_(next_word.item()).to(device)], dim=1)
        
        if next_word == eos_idx:
            break
        
    return decoder_input.squeeze(0)

def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
    model.eval()
    count = 0
    
    source_texts = []
    expected = []
    predicted = []
    
    # Size of the control window
    console_width = 80
    
    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch['encoder_input'].to(device) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(device) # (batch_size, seq_len)
            
            assert encoder_input.size(0) == 1 # Only one example at a time
            
            model_output = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_output_text = tokenizer_tgt.decode(model_output.detach().cpu().numpy())
            
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_output_text)
            
            # Print to console
            print_msg("-"*console_width)
            print_msg(f"Source: {source_text}")
            print_msg(f"Expected: {target_text}")
            print_msg(f"Predicted: {model_output_text}")
            
            if count == num_examples:
                break
            
        if writer:
            pass # TODO: TorchMetrics CharErrorRate, BLEU, WordErrorRate
            


def get_all_sentences(ds, lang):
    for example in ds:
        yield example['translation'][lang]

def get_or_build_tokenizer(config, ds, lang):
    # config['tokenizer_path'] = 'tokenizer_{}.json'
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            # vocab_size=config['vocab_size'],
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return tokenizer

def get_dataset(config):
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')
    
    # Build or load tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])
    
    # Keep 90% for training and 10% for validation
    train_size = int(len(ds_raw) * 0.9)
    val_size = len(ds_raw) - train_size
    ds_train_raw, ds_val_raw = random_split(ds_raw, [train_size, val_size])
    
    train_ds = BilingualDataset(ds_train_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    val_ds = BilingualDataset(ds_val_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'], config['seq_len'])
    
    max_len_src = 0
    max_len_tgt = 0
    
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))
        
    print("Max length of source language: ", max_len_src)
    print("Max length of target language: ", max_len_tgt)
    
    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=False)
    
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt

def get_model(config, vocabulary_src_len, vocab_tgt_len):
    model = build_transformer(vocabulary_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model']
                              )
    return model

def train_model(config):
    # Define device
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device: ", config['device'])
    
    # Make sure weights folder is created
    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)
    
    
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_dataset(config)
    model = get_model(config, len(tokenizer_src.get_vocab()), len(tokenizer_tgt.get_vocab())).to(config['device'])
    
    # Define writer for tensorboard
    writer = SummaryWriter(config['experiment_name'])
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-09)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer_tgt.token_to_id("[PAD]"), label_smoothing=0.1)
    
    
    initial_epoch = 0
    global_step = 0
    if config['preload'] is not None:
        model_filename = get_weights_file_path(config, config['preload'])
        print("Loading weights from: ", model_filename)
        state = torch.load(model_filename)
        initial_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        global_step = state['global_step']
        
    model.train()
    for epoch in range(initial_epoch, config['num_epochs']):
        print("Epoch: ", epoch)
        batch_iterator = tqdm(train_dataloader, total=len(train_dataloader), desc=f"Epoch {epoch}")
        for batch in batch_iterator:
            
            encoder_input = batch['encoder_input'].to(config['device']) # (batch_size, seq_len)
            decoder_input = batch['decoder_input'].to(config['device']) # (batch_size, seq_len)
            encoder_mask = batch['encoder_mask'].to(config['device']) # (batch_size, seq_len)
            decoder_mask = batch['decoder_mask'].to(config['device']) # (batch_size, seq_len)
            
            # print("Encoder input: ", encoder_input.dtype)
            # print("Decoder input: ", decoder_input.dtype)
            # print("Encoder mask: ", encoder_mask.dtype)
            # print("Decoder mask: ", decoder_mask.dtype)
            # exit()
            
            # Run the tensors through the model
            encoder_output = model.encode(encoder_input, encoder_mask) # (batch_size, seq_len, d_model)
            decoder_output = model.decode(decoder_input, encoder_output, encoder_mask, decoder_mask) # (batch_size, seq_len, d_model)
            projection_output = model.project(decoder_output) # (batch_size, seq_len, vocab_size)
            
            label = batch['label'].to(config['device']) # (batch_size, seq_len)
            loss = criterion(projection_output.view(-1, tokenizer_tgt.get_vocab_size()), label.view(-1))
            batch_iterator.set_postfix({"loss": loss.item()})
            
            # Log on tensorboard
            writer.add_scalar("Loss/train", loss.item(), global_step)
            writer.flush()
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            global_step += 1
            
        run_validation(model, val_dataloader, tokenizer_src, tokenizer_tgt, config['seq_len'], 
                           config['device'], lambda msg: batch_iterator.write(msg), global_step, writer)
            
        # Save model
        model_filename = get_weights_file_path(config, epoch)
        print("Saving model to: ", model_filename)
        torch.save({
            'epoch': epoch,
            'global_step': global_step,
            'optimizer': optimizer.state_dict(),
            'model_state_dict': model.state_dict()
        }, model_filename)
        
if __name__ == "__main__":
    config = get_config()
    train_model(config)
            
            
                    
