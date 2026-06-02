import numpy as np
import torch
import torch.nn as nn

class PCA:
    def __init__(self, k_components):
        self.k = k_components
        self.mean = None
        self.std = None
        self.components = None

    def fit_scaler(self, X):
        X = np.array(X, dtype=float)
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)

    def transform_scaler(self, X):
        X = np.array(X, dtype=float)
        std = np.where(np.isclose(self.std, 0), 1, self.std)
        return (X - self.mean) / std
    
    def fit_transform_scaler(self, X):
        self.fit_scaler(X)
        return self.transform_scaler(X)

    def fit_pca(self, X):
        cov_mat = np.cov(X, rowvar=False)
        eigvals, eigvects = np.linalg.eigh(cov_mat)

        order = np.argsort(eigvals)[::-1]
        self.components = eigvects[:, order]
        self.eigvals = eigvals[order]

    def transform_pca(self, X, k=None):
        pc = self.components[:, :k] if k else self.components[:, :self.k]
        return np.dot(X, pc)

    def fit_transform_pca(self, X):
        self.fit_pca(X)
        return self.transform_pca(X)
    
    def explained_variance_ratio(self):
        total = self.eigvals.sum()
        return self.eigvals / total

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, input_dim), nn.Sigmoid(),
        )

    def forward(self, X):
        return self.decoder(self.encoder(X))
    
    def encode(self, X):
        return self.encoder(X)
    
    def decode(self, z):
        return self.decoder(z)