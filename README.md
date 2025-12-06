# Whispersian-Serve
A simple FastAPI backend for transcribing Persian audio files into plain text using the onnx whisper model.

## Features
- Upload audio files for transcription
- Support for `.wav` audio format
- Returns transcription text as JSON

## Installation
1. Clone the repository:
```bash
git clone https://github.com/Parsa2201/Whispersian-Serve.git
cd Whispersian-Serve
```

2. Run the backend using docker:
```bash
docker compose up
```

## Usage
After running the background, it should be working on http://localhost:8000.
For a transcription request, send request in this format to the url http://localhost:8000/transcribe:
```bash
curl -X POST "http://127.0.0.1:8000/transcribe" -F "file=@path/to/audio.wav"
```
The response should be like this:
```json
{
    "text": "Your transcription."
}
```

## Change model
For using another model, copy the `encode.onnx` and `decoder.onnx` produced by the project Whispersian-Trainer to the directory `model/`. The backend then uses this new model for transcription.
- Warning: **DO NOT CHANGE THE `*.prototxt` FILES.**

## Test
For testing, run:
```bash
python -m pytest -v
```
If you change the model, add `test.csv` to directory `model/test` and write test rows to ensure that the onnx you have brought is the same as your actual fine-tuned model. Fill in the `model/test/audio` directory with your test `.wav` audios. In `test.csv` the name of those audio files (without `.wav` extension) with their transcription by the actual model should be inserted. The structure of the `test.csv` file should be:
```csv
filename,transcript
```