import os
from itertools import islice

import torch

from models.transformer import TransformerModel
from utils import create_encode, int_to_str, str_to_int

# hyperparameters
batch_size = 12  # how many independent sequences will we process in parallel?
block_size = 64  # what is the maximum context length for predictions?
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.0
max_iters = 1000
eval_interval = 125
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 20
# ------------

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("data/minishakespeare.txt", "r", encoding="utf-8") as f:
    lines = list(islice(f, 10000))
    text = "".join(lines)

# get unique characters text and create encoding/decoding mappings
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi, itos = str_to_int(chars), int_to_str(chars)
encode = create_encode(stoi)

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


hyperparameters = {
    "n_embd": n_embd,
    "n_head": n_head,
    "n_layer": n_layer,
    "block_size": block_size,
    "dropout": dropout,
    "vocab_size": vocab_size,
    "device": device,
}

model = TransformerModel(**hyperparameters)
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters()) / 1e6, "M parameters")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )
        checkpoint = {
            "model": model.state_dict(),
            "hyperparameters": hyperparameters,
            "optimizer": optimizer.state_dict(),
            "iter_num": iter,
            "stoi": stoi,
            "itos": itos,
        }
        torch.save(checkpoint, os.path.join(".checkpoints", "ckpt.pt"))

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
