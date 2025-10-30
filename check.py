import torch
from pathlib import Path

DATA_DIR = Path("/Users/watanabesaki/PycharmProjects/AARAN/data/processed")
train = torch.load(DATA_DIR / "hetero_graph_train.pt", weights_only=False)
print(train.keys())