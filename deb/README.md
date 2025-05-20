# ISKCON Translation Project

This project provides a machine translation system specifically designed for translating texts between English and spanish. It uses the Hugging Face Transformers library with a MarianMT model architecture.

## Features

- Translation from English to spanish
- Model retraining capabilities with new data
- Batch translation support
- Pre-trained model fine-tuning

## Setup

1. Install the required dependencies:
```bash
pip install transformers torch pandas
```

2. Ensure you have the pre-trained model in the project directory or specify its path in the configuration.

## Usage

### Translation

To translate text, use the `translate.py` script:

```python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Translate text
input_text = ["Your English text here"]
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
translated_tokens = model.generate(**inputs)
translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
```

### Retraining

To retrain the model with new data, use the `retrain.py` script:

1. Prepare your training data in a CSV/TXT file with tab-separated values containing:
   - English text
   - spanish text
   - (Optional) metadata

2. Update the file paths in `retrain.py`:
   - `new_file_path`: Path to your new training data
   - `pretrained_model_path`: Path to the current model
   - Set appropriate output directories for the retrained model

3. Run the retraining script:
```bash
python retrain.py
```

## Model Architecture

The project uses the MarianMT architecture from Hugging Face Transformers, which is specifically designed for neural machine translation. The model configuration includes:

- Encoder-decoder architecture
- Attention mechanism
- Vocabulary size: 65001 tokens
- Maximum sequence length: 512 tokens

## Training Parameters

The retraining process uses the following default parameters:

- Batch size: 20
- Number of epochs: 2
- Save checkpoints every 1500 steps
- Gradient checkpointing: disabled
- Learning rate scheduling: enabled

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]

## Contact

[Add your contact information here] 