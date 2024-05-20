from flask import Flask, request, jsonify
from speechbrain.inference.ASR import EncoderDecoderASR
import tempfile
import logging
import sys
import os

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("app.log"),
    logging.StreamHandler(sys.stdout)
])

try:
    print("Loading SpeechBrain model...")
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech")
    logging.info("SpeechBrain model loaded successfully.")
except Exception as e:
    logging.exception(f"Error loading SpeechBrain model: {str(e)}")
    sys.exit(1)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logging.info("Received request at /transcribe endpoint")

    if 'audio' not in request.files:
        logging.error("No audio file found in the request.")
        return jsonify({'error': 'No audio file provided.'}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        audio_file.save(temp_audio_file.name)
        logging.info(f"Saved audio file to temporary location: {temp_audio_file.name}")

        try:
            logging.info("Starting transcription...")
            text = asr_model.transcribe_file(temp_audio_file.name)
            logging.info(f"Transcription completed. Result: {text}")
        except Exception as e:
            logging.exception(f"Error during transcription: {str(e)}")
            return jsonify({'error': 'An error occurred during transcription.'}), 500
        finally:
            os.remove(temp_audio_file.name)

    return text  

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
