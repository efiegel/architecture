import os
from itertools import islice

import torch

from models.transformer import TransformerModel

device = "cpu"
ckpt_path = os.path.join(".checkpoints", "ckpt.pt")
checkpoint = torch.load(ckpt_path, map_location=device)

with open("data/minishakespeare.txt", "r", encoding="utf-8") as f:
    lines = list(islice(f, 10000))
    text = "".join(lines)

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

model = TransformerModel(vocab_size)
model.load_state_dict(checkpoint["model"])
m = model.to(device)

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
