from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
from TTS.api import TTS
import uuid
import os

app = Flask(__name__)
CORS(app)

voice_models = {
    'MARTIN-LUTHER-KING-JR': "voice_models/martin.wav",
    'ELON-MUSK': "voice_models/elon_musk.wav",
    #'ELON-LARGE': "voice_models/elon_MUSK_large.wav",
    'TAYLOR-SWIFT': "voice_models/taylor.wav",
}

# Initialize TTS at the start of the server
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)
os.makedirs("generated_speech", exist_ok=True)


@app.route('/tts', methods=['POST'])
def tts_endpoint():
    data = request.json
    text = data.get('text')
    voice_model = data.get('voice_model')
    if not text or not voice_model:
        return jsonify({"error": "Text and voice model are required"}), 400
    # voice_model = "voice_models/martin.wav"
    request_id = str(uuid.uuid4())
    print(f"Generating TTS for request_id: {request_id}")
    file_path = f"generated_speech/{request_id}.wav"
    print(f"Saving TTS to file_path: {file_path}")
    tts.tts_to_file(text=text, speaker_wav=voice_models[voice_model], language="en", file_path=file_path)

    # return jsonify({"message": "TTS generated successfully", "file_path": file_path})
    return send_file(file_path, mimetype='audio/wav')


@app.route('/list_voice_models', methods=['GET'])
def list_voice_models():
    models = tts.list_models()
    return jsonify({"models": models,"voices":voice_models})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)
