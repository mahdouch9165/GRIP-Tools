from flask import Flask, request, jsonify
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from rake_nltk import Rake

import sys

# Download the 'punkt' resource
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('wordnet')
nltk.download('vader_lexicon')
nltk.download('stopwords')

app = Flask(__name__)

@app.route('/tokenize', methods=['POST'])
def tokenize():
    # Get the text from the request
    text = request.json['text']

    try:
        # Tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)
        return jsonify({'tokens': tokens})
    except Exception as e:
        # Log the error message to the console
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/pos_tag', methods=['POST'])
def pos_tag():
    # Get the text from the request
    text = request.json['text']

    try:
        # Tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)
        # Perform part-of-speech tagging
        tagged = nltk.pos_tag(tokens)
        return jsonify({'tagged': tagged})
    except Exception as e:
        # Log the error message to the console
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/ner', methods=['POST'])
def named_entity_recognition():
    # Get the text from the request
    text = request.json['text']

    try:
        # Tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)
        # Perform part-of-speech tagging
        tagged = nltk.pos_tag(tokens)
        # Perform named entity recognition
        entities = nltk.ne_chunk(tagged)
        
        # Convert the named entities to a list of dictionaries
        entity_list = []
        for chunk in entities:
            if hasattr(chunk, 'label'):
                entity_list.append({'text': ' '.join(c[0] for c in chunk), 'label': chunk.label()})
            else:
                entity_list.append({'text': chunk[0], 'label': chunk[1]})
        
        return jsonify({'entities': entity_list})
    except Exception as e:
        # Log the error message to the console
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/lemmatize', methods=['POST'])
def lemmatize():
    # Get the text from the request
    text = request.json['text']

    try:
        # Tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)
        # Create a lemmatizer object
        lemmatizer = WordNetLemmatizer()
        # Perform lemmatization on each token
        lemmas = [lemmatizer.lemmatize(token) for token in tokens]
        return jsonify({'lemmas': lemmas})
    except Exception as e:
        # Log the error message to the console
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/stem', methods=['POST'])
def stem():
    # Get the text from the request
    text = request.json['text']

    try:
        # Tokenize the text using NLTK
        tokens = nltk.word_tokenize(text)
        # Create a stemmer object
        stemmer = PorterStemmer()
        # Perform stemming on each token
        stems = [stemmer.stem(token) for token in tokens]
        return jsonify({'stems': stems})
    except Exception as e:
        # Log the error message to the console
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/sentiment', methods=['POST'])
def sentiment_analysis():
    # Get the text from the request
    text = request.json['text']

    try:
        # Create a sentiment analyzer object
        analyzer = SentimentIntensityAnalyzer()
        # Perform sentiment analysis on the text
        sentiment_scores = analyzer.polarity_scores(text)
        return jsonify({'sentiment_scores': sentiment_scores})
    except Exception as e:
        # Log the error message to the console
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/keywords', methods=['POST'])
def keyword_extraction():
    # Get the text from the request
    text = request.json['text']

    try:
        # Create a Rake object
        r = Rake()

        # Extract keywords from the text
        r.extract_keywords_from_text(text)

        # Get the ranked phrases
        ranked_phrases = r.get_ranked_phrases_with_scores()

        return jsonify({'keywords': ranked_phrases})
    except Exception as e:
        # Log the error message to the console
        print(f"Error: {str(e)}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)