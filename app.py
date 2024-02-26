import math
import numpy as np
import joblib
from flask import Flask, request, jsonify

def count_characters(text):
    """
    Counts the total number of characters in the text, considering letters (uppercase and lowercase),
    hyphens, numbers, signs, and other symbols.
    
    Args:
    - text: The input text
    
    Returns:
    - num_characters: The total number of characters in the text
    """
    num_characters = 0
    
    # Define valid characters
    valid_characters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-0123456789!@#$%^&*()_+={[}]|\\:;'<,>.?/"
    
    for char in text:
        # Check if the character is a valid character
        if char in valid_characters:
            # Increment the count for each valid character
            num_characters += 1
    
    return num_characters

def count_words(text):
    """
    Counts the total number of words in the text.
    
    Args:
    - text: The input text
    
    Returns:
    - num_words: The total number of words in the text
    """
    num_words = 0
    
    # Define word delimiters
    word_delimiters = [' ', '\n']
    
    for i, char in enumerate(text):
        # Increment the count if the current character is a word delimiter and the previous character is not a delimiter or a hyphen
        if char in word_delimiters and text[i - 1] not in word_delimiters and text[i - 1] != '-':
            num_words += 1
    
    # Increment the count for the last word if the text doesn't end with a delimiter
    if ord(text[-1]) not in [ord(delimiter) for delimiter in word_delimiters]:
        num_words += 1
    
    return num_words

def count_sentences(text):
    """
    Counts the total number of sentences in the text.
    
    Args:
    - text: The input text
    
    Returns:
    - num_sentences: The total number of sentences in the text
    """
    num_sentences = 0
    
    # Define sentence delimiters
    sentence_delimiters = ['.', '!', '?', ';']
    
    for i, char in enumerate(text):
        # Increment the count if the current character is a sentence delimiter and the previous character is not a sentence delimiter
        if char in sentence_delimiters and text[i - 1] not in sentence_delimiters:
            num_sentences += 1
    
    return num_sentences

def count_syllables(word):
    """
    Counts the total number of syllables in the word.
    
    Args:
    - word: The input word
    
    Returns:
    - num_syllables: The total number of syllables in the word
    """
    num_syllables = 0
    
    # Define vowels, diphthongs, and triphthongs in Portuguese
    vowels = ['a', 'ã', 'â', 'á', 'à', 'e', 'é', 'ê', 'i', 'í', 'o', 'ô', 'õ', 'ó', 'u', 'ú']
    diphthongs = ['ãe', 'ai', 'ão', 'au', 'ei', 'eu', 'éu', 'ia', 'ie', 'io', 'iu', 'õe', 'oi', 'ói', 'ou', 'ua', 'ue', 'uê', 'ui']
    triphthongs = ['uai', 'uei', 'uão', 'uõe', 'uiu', 'uou']

    # Convert word to lowercase for easier comparison
    word = word.lower()
    
    # Counting syllables
    for i in range(len(word)):
        if word[i] in vowels:
            num_syllables += 1
            # Handling diphthongs
            if i > 0 and word[i-1:i+1] in diphthongs:
                num_syllables -= 1
            # Handling triphthongs
            if i > 1 and word[i-2:i+1] in triphthongs:
                num_syllables -= 1
    
    return num_syllables

def flesch_reading_ease(num_words, num_sentences, num_syllables):
    """
    Calculate the Flesch Reading Ease score.
    
    Args:
    - num_words: Total number of words in the text
    - num_sentences: Total number of sentences in the text
    - num_syllables: Total number of syllables in the text
    
    Returns:
    - score: Flesch Reading Ease score
    """
    score = 227 - 1.04 * (num_words / num_sentences) - 72 * (num_syllables / num_words)
    return score

def indice_gulpease(num_chars, num_words, num_sentences):
    """
    Calculate the Indice Gulpease.
    
    Args:
    - num_chars: Total number of characters in the text
    - num_words: Total number of words in the text
    - num_sentences: Total number of sentences in the text
    
    Returns:
    - indice: Indice Gulpease
    """
    indice = 89 + 300 * (num_sentences / num_words) - 10 * (num_chars / num_words)
    return indice

def coleman_liau(num_chars, num_words, num_sentences):
    """
    Calculate the Coleman-Liau index.
    
    Args:
    - num_chars: Total number of characters in the text
    - num_words: Total number of words in the text
    - num_sentences: Total number of sentences in the text
    
    Returns:
    - index: Coleman-Liau index
    """
    # Coleman-Liau index formula
    index = 5.4 * (num_chars / num_words) - 21 * (num_sentences / num_words) - 14
    return index

def automated_readability_index(num_chars, num_words, num_sentences):
    """
    Calculate the Automated Readability Index (ARI).
    
    Args:
    - num_chars: Total number of characters in the text
    - num_words: Total number of words in the text
    - num_sentences: Total number of sentences in the text
    
    Returns:
    - ari: Automated Readability Index
    """
    # Calculate ARI formula
    ari = 0.44 * (num_words / num_sentences) + 4.6 * (num_chars / num_words) - 20
    return ari

app = Flask(__name__)

# Load the trained model
model = joblib.load('/app/model/trained_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    # Extract the text from the request data
    text = data['excerpt']
    # Calculate readability features from the text
    num_chars = count_characters(text)
    num_words = count_words(text)
    num_sentences = count_sentences(text)
    num_syllables = count_syllables(text)  # You need to implement this function
    flesch = flesch_reading_ease(num_words, num_sentences, num_syllables)
    gulpease = indice_gulpease(num_chars, num_words, num_sentences)
    colemanliau = coleman_liau(num_chars, num_words, num_sentences)
    ari = automated_readability_index(num_chars, num_words, num_sentences)
    features = np.array([flesch, gulpease, colemanliau, ari]).reshape(1, -1)
    # Make prediction
    probabilities = model.predict_proba(features)
    # Probability of being 1
    probability = round(probabilities[0][1] * 100,2)
    
    # Construct dictionary containing prediction and feature values
    response = {
        'probability': probability,
        'features': {
            'flesch': round(flesch,2),
            'gulpease': round(gulpease,2),
            'colemanLiau': round(colemanliau,2),
            'ari': round(ari,2)
        }
    }
    # Return the response as JSON
    return jsonify(response)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)

