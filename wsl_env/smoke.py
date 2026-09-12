import sys, os
sys.path.insert(0, os.path.expanduser("~/projects/Matcha-TTS"))

import torch
print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())

import matcha
from matcha.models.components.flow_matching import CFM
from matcha.models.components.text_encoder import TextEncoder
from matcha.models import matcha_tts
print("MATCHA_IMPORT_OK")
