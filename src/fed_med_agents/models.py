import torch
from torch import nn


class Classifier(nn.Module):
    def __init__(self, inputs):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(inputs, 24), nn.ReLU(), nn.Linear(24, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def shared_model():
    return Classifier(32)


def train(model, x, y, epochs=10, seed=42):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(x), torch.tensor(y)),
        batch_size=32,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = nn.functional.binary_cross_entropy_with_logits(model(xb), yb)
            loss.backward()
            optimizer.step()
    return float(loss.detach())


def probabilities(model, x):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(x))).numpy()
