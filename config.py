import os

def get_config():
    return {
        "batch_size": 32,
        "num_epochs": 10,
        "lr": 0.0001,
        "seq_len": 350,
        "d_model": 512,
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_filename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizer_{}.json",
        "experiment_name": "runs/tmodel_",
    }
    
def get_weights_file_path(config, epoch: str):
    model_folder = config['model_folder']
    model_basename = config['model_filename']
    model_filename = f"{model_basename}{epoch}.pt"
    return os.path.join(model_folder, model_filename)