import os
from dotenv import load_dotenv


load_dotenv()

SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE"))

# DEVICE = "cpu"
# if torch.cuda.is_available():
#     DEVICE = "gpu"
# elif torch.backends.mps.is_available():
#     DEVICE = "mps"
# DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

MODEL_TYPE = os.environ.get("MODEL_TYPE")
MODEL_PATH = os.environ.get("MODEL_PATH")

SAMPLE_AUDIO = os.environ.get("SAMPLE_AUDIO")