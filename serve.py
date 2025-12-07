import os
from fastapi import FastAPI, File, UploadFile
from src.transcribe.onnx_trans import OnnxTrans
from src.config import *

app = FastAPI()

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile):
    audio_bytes = await file.read()

    if not os.path.exists("data"):
        os.makedirs("data")
    audio_path = f"data/{file.filename}"
    with open(audio_path, "wb") as f_out:
        f_out.write(audio_bytes)

    model = OnnxTrans(model_path=MODEL_PATH, model_type=MODEL_TYPE)
    text = model.transcribe([audio_path])[0]

    return {"text": text}