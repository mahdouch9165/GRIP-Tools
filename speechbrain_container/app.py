from flask import Flask, request, jsonify, send_file
from speechbrain.inference.ASR import EncoderDecoderASR
from speechbrain.inference.separation import SepformerSeparation as separator
from speechbrain.inference.TTS import Tacotron2
from speechbrain.inference.vocoders import HIFIGAN
import nltk
import io
import zipfile
import torch
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
            est_sources = sep_model.separate_file(temp_audio_file.name)
            logging.info("Speech separation completed.")
        except Exception as e:
            logging.exception(f"Error during speech separation: {str(e)}")
            return jsonify({'error': 'An error occurred during speech separation.'}), 500
        finally:
            os.remove(temp_audio_file.name)

    # Save the separated sources as WAV files
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zip_file:
        for i in range(est_sources.shape[2]):
            source_filename = f"source{i+1}.wav"
            source_data = est_sources[:, :, i].detach().cpu()
            torchaudio.save(source_filename, source_data, 8000)
            zip_file.write(source_filename)
            os.remove(source_filename)

    zip_buffer.seek(0)

    return send_file(
        zip_buffer,
        mimetype='application/zip',
        as_attachment=True,
        download_name='separated_sources.zip'
    )

@app.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    logging.info("Received request at /text-to-speech endpoint")
    
    if 'text' not in request.files:
        logging.error("No text file found in the request.")
        return jsonify({'error': 'No text file provided.'}), 400

    text_file = request.files['text']
    text_content = text_file.read().decode('utf-8')
    
    # try loading nltk punkt
    try:
        print("Loading NLTK punkt model...")
        nltk.download('punkt')
        def segment_text(text):
            sentences = nltk.sent_tokenize(text)
            return sentences
        logging.info("NLTK punkt model loaded successfully.")
    except Exception as e:
        logging.exception(f"Error loading SpeechBrain model: {str(e)}")
        sys.exit(1)
        
    sentences = segment_text(text_content)

    # Create tuples of (index, sentence) and sort based on sentence length
    indexed_sentences = sorted(enumerate(sentences), key=lambda x: len(x[1]), reverse=True)

    try:
        logging.info("Loading Tacotron2 TTS model...")
        tmpdir_tts = os.path.join(os.getcwd(), "tmpdir_tts")
        os.makedirs(tmpdir_tts, exist_ok=True)
        tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir=tmpdir_tts)
        logging.info("Tacotron2 TTS model loaded successfully.")
    except Exception as e:
        logging.exception(f"Error loading Tacotron2 TTS model: {str(e)}")
        return jsonify({'error': 'An error occurred while loading the TTS model.'}), 500

    try:
        logging.info("Loading HIFIGAN vocoder...")
        tmpdir_vocoder = os.path.join(os.getcwd(), "tmpdir_vocoder")
        os.makedirs(tmpdir_vocoder, exist_ok=True)
        hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir=tmpdir_vocoder)
        logging.info("HIFIGAN vocoder loaded successfully.")
    except Exception as e:
        logging.exception(f"Error loading HIFIGAN vocoder: {str(e)}")
        return jsonify({'error': 'An error occurred while loading the vocoder.'}), 500

    try:
        logging.info("Starting text-to-speech...")
        # Extract the sorted sentences from the indexed tuples
        sorted_sentences = [sentence for _, sentence in indexed_sentences]
        mel_outputs, mel_lengths, alignments = tacotron2.encode_batch(sorted_sentences)
        logging.info("Text-to-speech completed.")
    except Exception as e:
        logging.exception(f"Error during text-to-speech: {str(e)}")
        return jsonify({'error': 'An error occurred during text-to-speech.'}), 500

    try:
        logging.info("Starting vocoder processing...")
        waveforms = hifi_gan.decode_batch(mel_outputs)
        logging.info("Vocoder processing completed.")
    except Exception as e:
        logging.exception(f"Error during vocoder processing: {str(e)}")
        return jsonify({'error': 'An error occurred during vocoder processing.'}), 500

    # Create a dictionary to map the original index to the generated waveform
    waveform_dict = {index: waveform for (index, _), waveform in zip(indexed_sentences, waveforms)}

    # Reorder the waveforms based on the original index
    reordered_waveforms = [waveform_dict[index] for index in range(len(sentences))]

    # Concatenate the reordered waveforms along the time dimension
    concatenated_waveform = torch.cat(reordered_waveforms, dim=1)

    # Convert the concatenated waveform to the desired sample format and channel layout
    concatenated_waveform = concatenated_waveform.squeeze(0)  # Remove the batch dimension
    concatenated_waveform = concatenated_waveform.unsqueeze(0)  # Add a channel dimension
    concatenated_waveform = concatenated_waveform.to(torch.float32)  # Convert to float32

    # Save the generated audio as a WAV file
    audio_bytes = io.BytesIO()
    torchaudio.save(audio_bytes, concatenated_waveform, 22050, format='wav')
    audio_bytes.seek(0)

    return send_file(
        audio_bytes,
        mimetype='audio/wav',
        as_attachment=True,
        download_name='tts_result.wav'
    )

if __name__ == '__main__':
    logging.info("Starting Flask application...")
    app.run(host='0.0.0.0', port=5000)
