## LOAD THE 'NEW' LANGUAGE TRANSLATION DATA.
import pandas as pd

# Load your translation data for training in .txt or .csv format
new_file_path = 'link to your file'  # Path to new training data
new_data = pd.read_csv(new_file_path, sep='\t', header=None, names=['english', 'spanish', 'metadata'])

# Drop any existing 'metadata' column
new_data = new_data.drop(columns=['metadata'], errors='ignore')

# Columns to lists conversion
english_sentences = new_data['english'].tolist()
spanish_sentences = new_data['spanish'].tolist()

# Need to be in string format
spanish_sentences = [str(sentence) for sentence in spanish_sentences]

## LOAD THE MODEL AND TOKENIZER FOR TRAINING/RETRAINING.

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
pretrained_model_path = "      link to folder 'currentModel'    "
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)

## PROCESS THE TRAINING DATA.

# Tokenize inputs and labels for the new data
inputs = tokenizer(english_sentences, return_tensors="pt", padding=True, truncation=True)
labels = tokenizer(spanish_sentences, return_tensors="pt", padding=True, truncation=True).input_ids

# Set padding tokens to -100 to ignore them in training
labels[labels == tokenizer.pad_token_id] = -100

# Custom dataset class for the new data
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Prepare the dataset for retraining
new_dataset = TranslationDataset(inputs, labels)

## SET THE TRAINING CONDITION.

from transformers import Trainer, TrainingArguments
import torch  # Import PyTorch
from transformers import TrainerCallback
import os
os.environ["WANDB_DISABLED"] = "true" # avoid W&B tracking experiments

# Define retraining arguments
retraining_args = TrainingArguments(
    output_dir="/content/drive/My Drive/translation_checkpoints_retrained",  # Save retrained model here
    num_train_epochs=2,  # Number of epochs for retraining
    per_device_train_batch_size=20,
    save_steps=1500,
    save_total_limit=1,
    logging_dir="/content/drive/My Drive/translation_checkpoints_retrained/logs",  # Log directory
)

# Define a custom callback to save the tokenizer with every checkpoint
class SaveTokenizerCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        checkpoint_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        if os.path.exists(checkpoint_dir):
            tokenizer.save_pretrained(checkpoint_dir)

# Custom dataset class
class TranslationDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)
# Prepare the dataset
dataset = TranslationDataset(inputs, labels)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=retraining_args,
    train_dataset=dataset,
    callbacks=[SaveTokenizerCallback()]  # Add the custom callback here
)

## TRAIN!
trainer.train()

## SAVE THE RETRAINED MODEL AND TOKENIZER
model.save_pretrained(" give the destination folder location")
tokenizer.save_pretrained("  give the destination folder location  ")
