import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt

def make_loder(X, batch_size, shuffle):
    if hasattr(X, "to_numpy"):
        X = X.to_numpy()
    t = torch.tensor(X, dtype=torch.float32)
    return DataLoader(TensorDataset(t), batch_size=batch_size, shuffle=shuffle)

def train_AE(model, X_train, X_test, device, epochs=50, lr=1e-3, batch_size=256):
    train_loader = make_loder(X_train, batch_size=batch_size, shuffle=True)
    test_loader = make_loder(X_test, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_losses, test_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        tl = 0
        for (x,) in train_loader:
            x = x.to(device)
            optimizer.zero_grad()
            loss = F.mse_loss(model(x), x)
            loss.backward()
            optimizer.step()
            tl += loss.item()

        model.eval()
        vl = 0
        with torch.no_grad():
            for (x,) in test_loader:
                x = x.to(device)
                vl += F.mse_loss(model(x), x).item()
        
        train_losses.append(tl / len(train_loader))
        test_losses.append(vl / len(test_loader))
    
        if epoch % 10 == 0:
            print(f"Epoch {epoch:>3}/{epochs}  train={train_losses[-1]:.4f}  test={test_losses[-1]:.4f}")

    return train_losses, test_losses

def plot_loss(train_losses, test_losses):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(train_losses, label="Train")
    ax.plot(test_losses,  label="Test")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Curva de entrenamiento — Autoencoder", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

def encode_dataset(model, X, device):
    """Obtiene embeddings para todo X. Usado en 2e."""
    model.eval()
    with torch.no_grad():
        if hasattr(X, "to_numpy"):
            X = X.to_numpy()
        t = torch.tensor(X, dtype=torch.float32).to(device)
        return model.encode(t).cpu().numpy()


def decode_dataset(model, Z, device):
    """Decodifica embeddings Z → imágenes reconstruidas. Usado en 2e."""
    model.eval()
    with torch.no_grad():
        if hasattr(Z, "to_numpy"):
            Z = Z.to_numpy()
        t = torch.tensor(Z, dtype=torch.float32).to(device)
        return model.decode(t).cpu().numpy()
