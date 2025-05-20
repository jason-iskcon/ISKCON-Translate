## TRANSLATION USING TRAINED MODEL
import argparse
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

def translate_text(input_text, model_path):
    """
    Translate the given text using the specified model.
    
    Args:
        input_text (list): List of strings to translate
        model_path (str): Path to the pretrained model
    
    Returns:
        list: List of translated texts
    """
    try:
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Tokenize the input
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
        # Generate translations
        translated_tokens = model.generate(**inputs)
        # Decode translations
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
        return translated_text

    except Exception as e:
        print(f"Error during translation: {str(e)}")
        return None

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Translate English text to spanish using a trained model.')
    parser.add_argument('--model_path', type=str, required=True,
                      help='Path to the pretrained model directory')
    parser.add_argument('--text', type=str, nargs='+', required=True,
                      help='Text to translate. Can be multiple sentences.')
    
    # Parse arguments
    args = parser.parse_args()    
    # Perform translation
    translations = translate_text(args.text, args.model_path)
    if translations:
        # Print results
        for original, translation in zip(args.text, translations):
            print(f"Original: {original}")
            print(f"Translation: {translation}")
            print("-" * 50)

if __name__ == "__main__":
    main()
