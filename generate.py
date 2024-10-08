import os

import torch

from models.transformer import TransformerModel
from utils import create_decode, get_device

device = get_device()
ckpt_path = os.path.join(".checkpoints", "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

stoi, itos = checkpoint["stoi"], checkpoint["itos"]
decode = create_decode(itos)

model = TransformerModel(**checkpoint["hyperparameters"])
model.load_state_dict(checkpoint["model"])
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
