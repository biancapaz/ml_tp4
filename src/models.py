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
        pc = self.components[:, :self.k] if k is None else self.components[:, :k]
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
    
class KMeans:
    def __init__(self, k=15, tol=1e-4, max_iter=500):
        self.k = k
        self.max_iterations = max_iter
        self.tolerance = tol
        self.centroids = None
        self.labels = None

    def init_centroids(self, X, random_state=42):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), self.k, replace=False)
        self.centroids = X[idx].copy()

    def compute_distances(self, X):
        N = len(X)
        dists = np.zeros((N, self.k))        # matriz vacía (N, K)

        for j in range(self.k):
            diff = X - self.centroids[j]     # (N, D) - (D,) → (N, D)
            dists[:, j] = (diff ** 2).sum(axis=1)  # distancia de todos los puntos al centroide j

        self.labels = dists.argmin(axis=1)   # (N,) → centroide más cercano por punto

    def update_centroids(self, X):
        for k in range(self.k):
            self.centroids[k] = X[self.labels == k].mean(axis=0)

    def compute_inertia(self, X):
        inertia = 0
        for k in range(self.k):
            pts = X[self.labels == k]
            inertia += ((pts - self.centroids[k]) ** 2).sum()
        return inertia

    def fit(self, X):
        self.init_centroids(X)

        for _ in range(self.max_iterations):
            centroids_old = self.centroids.copy()
            
            self.compute_distances(X)
            self.update_centroids(X)

            shift = np.linalg.norm(self.centroids - centroids_old)
            if shift < self.tolerance:
                break
        
        self.inertia = self.compute_inertia(X)

class GMM:
    def __init__(self, k, tol=1e-3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.tol  = tol
        self.pi = None
        self.mu = None
        self.sigma = None
        self.log_likelihoods = []

    def init_params(self, X):
        N, D = X.shape
        km = KMeans(k=self.k)
        km.fit(X)

        self.mu = km.centroids.copy()
        self.pi = np.ones(self.k) / self.k
        self.sigma = np.full((self.k, D), X.var(axis=0))
    
    def log_gaussian(self, X):
        """
        Devuelve log N(x | mu_k, sigma_k) para cada punto y cada cluster.
        Shape resultado: (N, K)
        """
        N, D = X.shape
        log_probs = np.zeros((N, self.k))
        
        for k in range(self.k):
            # diferencia de cada punto con la media del cluster k
            diff = X - self.mu[k]                          # (N, D)
            
            # término cuadrático: suma sobre dimensiones de (x-mu)²/sigma
            quad = ((diff ** 2) / self.sigma[k]).sum(axis=1)  # (N,)
            
            # término logarítmico: suma de log(sigma) por dimensión
            log_det = np.log(self.sigma[k]).sum()           # escalar
            
            # log N(x|mu,sigma) = -D/2 log(2π) - 1/2 log|Σ| - 1/2 (x-μ)ᵀΣ⁻¹(x-μ)
            log_probs[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + quad)
        
        return log_probs  # (N, K)

    def e_step(self, X):
        """
        Calcula r[n,k] = P(cluster k | punto n)
        Shape: (N, K)
        """
        # log pi_k + log N(x_n | mu_k, sigma_k) para cada n,k
        log_probs = self.log_gaussian(X) + np.log(self.pi)  # (N, K)
        
        # logsumexp: log(sum(exp(x))) de forma estable
        # para cada punto, el denominador de la responsabilidad
        log_sum = np.log(np.exp(log_probs - log_probs.max(axis=1, keepdims=True)).sum(axis=1, keepdims=True))
        log_sum += log_probs.max(axis=1, keepdims=True)
        
        # responsabilidades en escala normal (no log)
        r = np.exp(log_probs - log_sum)   # (N, K)
        return r
    
    def m_step(self, X, r):
        """
        Actualiza pi, mu, sigma usando las responsabilidades r.
        """
        N, D = X.shape
        
        # cantidad efectiva de puntos en cada cluster
        Nk = r.sum(axis=0)                    # (K,) — suma de responsabilidades por cluster
        
        # nuevos pesos: proporción del dataset que representa cada cluster
        self.pi = Nk / N                      # (K,)
        
        # nuevas medias: promedio ponderado por responsabilidades
        self.mu = (r.T @ X) / Nk[:, np.newaxis]   # (K, D)
        # r.T @ X → (K, N) @ (N, D) = (K, D) → suma ponderada por cluster
        
        # nuevas varianzas: varianza ponderada por dimensión
        for k in range(self.k):
            diff = X - self.mu[k]                          # (N, D)
            # varianza diagonal ponderada
            self.sigma[k] = (r[:, k:k+1] * diff ** 2).sum(axis=0) / Nk[k]  # (D,)
            # clampear para evitar varianzas que colapsen a 0
            self.sigma[k] = np.maximum(self.sigma[k], 1e-6)

    def log_likelihood(self, X):
        """
        Log-verosimilitud del modelo dado X.
        Mide qué tan bien explica el modelo los datos — sube con cada iteración EM.
        """
        log_probs = self.log_gaussian(X) + np.log(self.pi)  # (N, K)
        
        # logsumexp por punto → log p(x_n)
        max_lp = log_probs.max(axis=1, keepdims=True)
        log_px = np.log(np.exp(log_probs - max_lp).sum(axis=1)) + max_lp.squeeze()  # (N,)
        
        return log_px.sum()   # escalar

    def fit(self, X):
        self.init_params(X)
        self.log_likelihoods = []
        
        for i in range(self.max_iter):
            # E-step
            r = self.e_step(X)
            
            # M-step
            self.m_step(X, r)
            
            # log-likelihood actual
            ll = self.log_likelihood(X)
            self.log_likelihoods.append(ll)
            
            # convergencia: mejora menor que tol
            if i > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                print(f"Convergió en iteración {i+1}")
                break
        
        return self

    def predict(self, X):
        """Asignación dura: argmax de responsabilidades."""
        r = self.e_step(X)
        return r.argmax(axis=1)   # (N,)