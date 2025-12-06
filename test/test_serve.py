import pytest
from fastapi.testclient import TestClient
from serve import app
import warnings
import pandas as pd
from src.transcribe.onnx_trans import OnnxTrans
from src.config import *

client = TestClient(app)

def test_transcription_module_structure():
    model = OnnxTrans(model_path='./model')
    assert type(model.transcribe([SAMPLE_AUDIO])[0]) == str

def test_transcription_module_validity():
    test_csv_dir = "model/test"
    test_csv_path = "model/test/test.csv"
    test_audio_dir = "model/test/audio"

    if not os.path.exists(test_csv_dir) or not os.path.exists(test_csv_path):
        # warnings.warn("Warning: there is no data test inside model/test.")
        pytest.skip("No data test 'model/test/test.csv'")
    
    df = pd.read_csv(test_csv_path)
    model = OnnxTrans(model_path='./model')

    for _, row in df.iterrows():
        audio_file = os.path.join(test_audio_dir, f"{row["filename"]}.wav")
        assert os.path.exists(audio_file)

        transcription = model.transcribe([audio_file])[0]
        assert transcription == row['transcript'], (
            f"Transcription mismatch for {row['filename']}: "
            f"expected '{row['transcript']}', got '{transcription}'"
        )

def test_transcription_api_status_code():
    with open(SAMPLE_AUDIO, "rb") as f:
        response = client.post("transcribe", files={'file': f})

    assert response.status_code == 200

def test_transcription_api_output_structure():
    with open(SAMPLE_AUDIO, "rb") as f:
        response = client.post("transcribe", files={'file': f})

    data = response.json()
    assert "text" in data