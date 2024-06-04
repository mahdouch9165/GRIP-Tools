from flask import Flask, request, jsonify
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.separation import SepformerSeparation as separator
import torchaudio
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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logging.info("Received request at /transcribe endpoint")

    try:
        print("Loading SpeechBrain model...")
        asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech")
        logging.info("SpeechBrain model loaded successfully.")
    except Exception as e:
        logging.exception(f"Error loading SpeechBrain model: {str(e)}")
        sys.exit(1)

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

@app.route('/separate_speech', methods=['POST'])
def separate_speech():
    logging.info("Received request at /separate_speech endpoint")

    try:
        print("Loading SpeechBrain model...")
        sep_model = separator.from_hparams(source="speechbrain/sepformer-wsj02mix")
        logging.info("SpeechBrain model loaded successfully.")
    except Exception as e:
        logging.exception(f"Error loading SpeechBrain model: {str(e)}")
        sys.exit(1)

    if 'audio' not in request.files:
        logging.error("No audio file found in the request.")
        return jsonify({'error': 'No audio file provided.'}), 400

    audio_file = request.files['audio']

    with tempfile.NamedTemporaryFile(delete=False) as temp_audio_file:
        audio_file.save(temp_audio_file.name)
        logging.info(f"Saved audio file to temporary location: {temp_audio_file.name}")

        try:
            logging.info("Starting speech separation...")
            sources = sep_model.separate_file(temp_audio_file.name)
            logging.info("Speech separation completed.")
        except Exception as e:
            logging.exception(f"Error during speech separation: {str(e)}")
            return jsonify({'error': 'An error occurred during speech separation.'}), 500
        finally:
            os.remove(temp_audio_file.name)

    return jsonify(sources)

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
