from flask import Flask, request, jsonify
import nltk
import sys

# Download the 'punkt' resource
nltk.download('punkt')

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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)