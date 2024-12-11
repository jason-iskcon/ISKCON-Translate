## TRANSLATION USING TRAINED MODEL

## STEP 1: set-up the model
# load the utilities
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# link the trained model and tokenizer
pretrained_model_path = "  link to the folder 'currentModel'   "

# Load model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_path)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_path)


## STEP 2: translate
# Type sentences to translate (e.g., English to French)
input_text = ["Good Morning everybody.", "Welcome to this session.", "Hope you are feeling good today.", "Now, we are going to start the discussion."]


# Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)

# Let the model generate translated tokens
translated_tokens = model.generate(**inputs)

# tokens to text decoding (final translated text)
translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

# lets see translated texts
for original, translation in zip(input_text, translated_text):
    print(f"Original: {original}")
    print(f"Translation: {translation}")
    print("-" * 50)

