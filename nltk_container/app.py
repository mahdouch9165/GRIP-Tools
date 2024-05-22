from flask import Flask, request, jsonify
import nltk

import sys

# Download the 'punkt' resource
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)