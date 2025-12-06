# Whispersian-Serve

# 1. Installation
Install the dependencies using the requirements.txt.
```bash
pip install -r requirements.txt
```

# 2. Usage
At this moment, the usable files are:
```
serve.py
```

## Serve the onnx
For testing if the fine-tuned onnx model is working, run the `serve.py` with uvicorn.
```bash
python -m uvicorn serve:app
```
Send a request in this format (the file should be in `.wav` format with 16000 sample rate):
```json
{
    "file": File
}
```
It will send the transcription after a while in this format:
```json
{
    "text": Transcription
}
```