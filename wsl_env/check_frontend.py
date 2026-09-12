import torch
from matcha.text.cleaners import english_cleaners2
from matcha.utils.monotonic_align import maximum_path
import numpy as np

s = english_cleaners2("Hello world, this is a Linux migration test!")
print("CLEANERS_OK:", s[:60])

val = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
mask = torch.tensor([[[1.0, 1.0, 1.0]]])
path = maximum_path(val, mask)
print("MONO_ALIGN_OK:", path.shape)
