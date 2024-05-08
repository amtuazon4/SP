import nltk
from nltk.corpus import wordnet
import random

nltk.download('wordnet')

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return list(synonyms)

def paraphrase_sentence(sentence):
    words = nltk.word_tokenize(sentence)
    paraphrased_words = []
    for word in words:
        synonyms = get_synonyms(word)
        if synonyms:
            paraphrased_words.append(random.choice(synonyms))
        else:
            paraphrased_words.append(word)
    return ' '.join(paraphrased_words)

# Example sentence to paraphrase
sentence = "The quick brown fox jumps over the lazy dog."

# Paraphrase the sentence
paraphrased_sentence = paraphrase_sentence(sentence)
print("Original sentence:", sentence)
print("Paraphrased sentence:", paraphrased_sentence)