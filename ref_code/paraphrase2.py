from genienlp.models import Seq2SeqModel

# Load pre-trained GenieNLP model
model_name = "genienlp/mt5-small-t2t_paraphrase"
model = Seq2SeqModel.from_pretrained(model_name)

def paraphrase_text(text):
    # Generate paraphrased text
    paraphrased_text = model.generate(text, max_length=100, num_return_sequences=1)
    
    # Return paraphrased text
    return paraphrased_text[0]['generated_text']

# Example input text to paraphrase
input_text = "The quick brown fox jumps over the lazy dog."

# Paraphrase the input text using GenieNLP
paraphrased_text = paraphrase_text(input_text)
print("Original text:", input_text)
print("Paraphrased text:", paraphrased_text)