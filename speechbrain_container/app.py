from flask import Flask, request, jsonify
from speechbrain.inference.ASR import EncoderDecoderASR
import torchaudio
import sys
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("app.log"),  # Log to a file named app.log
    logging.StreamHandler(sys.stdout)  # Log to the console (stdout)
])

try:
    # Load the SpeechBrain EncoderDecoderASR model
    logging.info("Loading SpeechBrain model...")
    asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-transformer-transformerlm-librispeech")
    logging.info("SpeechBrain model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading SpeechBrain model: {str(e)}")
    sys.exit(1)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logging.info("Received request at /transcribe endpoint")
    
    if 'audio' not in request.files:
        logging.error("No audio file found in the request.")
        return jsonify({'error': 'No audio file provided.'}), 400
    
    audio_file = request.files['audio']
    logging.info(f"Received audio file: {audio_file.filename}")
    
    try:
        logging.info("Starting transcription...")
        text = asr_model.transcribe_file(audio_file)
        logging.info(f"Transcription completed. Result: {text}")
        return text
    except Exception as e:
        logging.error(f"Error during transcription: {str(e)}")
        return jsonify({'error': 'An error occurred during transcription.'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)