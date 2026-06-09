import numpy as np
import torch
import torch.nn as nn

class PCA:
    def __init__(self, k_components):
        self.k = k_components
        self.mean = None
        self.components = None
        self.eigvals = None

    def fit_scaler(self, X):
        """ Aprende la media de X_train"""
        X = np.array(X, dtype=float)
        self.mean = X.mean(axis=0)

    def transform_scaler(self, X):
        """Centra usando la media ya aprendida"""
        X = np.array(X, dtype=float)
        return X - self.mean
    
    def fit_transform_scaler(self, X): 
        """Shortcut para X_train: aprende y centra en un paso"""
        self.fit_scaler(X)
        return self.transform_scaler(X)

    def fit_pca(self, X):
        """Aprende componentes principales.
        - Espera X sin escalado --> solo usa transform_scaler"""
        X_scaled = self.transform_scaler(X)

        cov_mat = np.cov(X_scaled, rowvar=False)
        eigvals, eigvects = np.linalg.eigh(cov_mat) # devuelve de menor a mayor

        self.components = eigvects[:, ::-1] # ivierto orden de las columnas
        self.eigvals = eigvals[::-1] # invierto el orden

    def transform_pca(self, X):
        """Proyecta X sin escalar al subespacion de k componentes"""
        if self.components is None:
            raise RuntimeError("Llama fit_pca antes de transform_pca")
        X_scaled = self.transform_scaler(X)
        pc = self.components[:, :self.k]
        return np.dot(X_scaled, pc)

    def fit_transform_pca(self, X): 
        """Shortcut para X_train: aprende componentes y proyecta en un mismo paso"""
        self.fit_pca(X)
        return self.transform_pca(X)
    
    def reconstruct_x_hat(self, Z):
        """Reconstruye X desde la representacion reducida Z"""
        pc = self.components[:, :self.k]
        X_hat = np.dot(Z, pc.T) # (N x k) x (k x D) -> (N x D)
        return X_hat + self.mean
    
    def explained_variance_ratio(self):
        total = self.eigvals.sum()
        return self.eigvals / total

class AE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, 1024), nn.ReLU(),
            nn.Linear(1024, input_dim), nn.Sigmoid(),
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
        self.inertia = None #distorsion

    def init_centroids(self, X, random_state=0):
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=self.k, replace=False)
        self.centroids = X[idx].copy()

    def _assign(self, X):
        # (N, 1, D) - (1, K, D) → (N, K, D) → (N, K)
        dist = ((X[:, None, :] - self.centroids[None, :, :]) ** 2).sum(axis=-1)
        self.labels = dist.argmin(axis=1)
        return dist

    def _update_centroids(self, X):
        for k in range(self.k):
            pts = X[self.labels == k]
            if len(pts) > 0:
                self.centroids[k] = pts.mean(axis=0)
            # si está vacío, el centroide no se mueve

    def fit(self, X, seed=0):
        self.init_centroids(X, seed)
        for _ in range(self.max_iterations):
            centroids_old = self.centroids.copy()
            dist = self._assign(X)
            self._update_centroids(X)

            shift = np.linalg.norm(self.centroids - centroids_old)
            if shift < self.tolerance:
                break
        
        self.inertia = dist[np.arange(len(X)), self.labels].sum()
        return self
    
    def predict(self, X):
        self._assign(X)
        return self.labels

class GMM:
    def __init__(self, k, tol=1e-3, max_iter=100):
        self.k = k
        self.max_iter = max_iter
        self.tol  = tol
        self.pi = None
        self.mu = None
        self.sigma = None
        self.log_likelihoods = []

    def init_params(self, X, km=None):
        N, D = X.shape
        if km is None:
            km = KMeans(k=self.k).fit(X)

        self.mu = km.centroids.copy()
        self.pi = np.ones(self.k) / self.k
        self.sigma = np.array([
            X[km.labels == k].var(axis=0).clip(1e-6, None)
            if (km.labels == k).any() else X.var(axis=0)
            for k in range(self.k)
        ])

    def _gaussian(self, X, k):
        """
        Densidad N(x | mu_k, sigma_k) para todos los puntos.
        Covarianza diagonal: sigma_k es un vector (D,), no una matriz.
        Devuelve (N,)
        """
        N, D = X.shape
        diff = X - self.mu[k]
        quad = ((diff ** 2) / self.sigma[k]).sum(axis=1)
        log_det = np.log(self.sigma[k]).sum()
        log_p = -0.5 * (D * np.log(2 * np.pi) + log_det + quad)
        return np.exp(log_p)   

    def e_step(self, X):
        """
        Calcula r[n,k] = P(cluster k | punto n)
        Shape: (N, K)
        """
        N = X.shape[0]
        r = np.zeros((N, self.k))

        for k in range(self.k):
            r[:, k] = self.pi[k] * self._gaussian(X, k)

        r /= r.sum(axis=1, keepdims=True)
        return r
    
    def m_step(self, X, r):
        """
        Actualiza pi, mu, sigma usando las responsabilidades r.
        """
        N, D = X.shape
        Nk = r.sum(axis=0)
        
        self.pi = Nk / N        
        self.mu = (r.T @ X) / Nk[:, np.newaxis]

        for k in range(self.k):
            diff = X - self.mu[k]
            self.sigma[k] = (r[:, k:k+1] * diff ** 2).sum(axis=0) / Nk[k]
            self.sigma[k] = self.sigma[k].clip(1e-6, None) # evita colapso a 0

    def log_likelihood(self, X):
        """
        Log-verosimilitud del modelo dado X.
        Mide qué tan bien explica el modelo los datos — sube con cada iteración EM.
        """
        N = X.shape[0]
        px = np.zeros(N)
        for k in range(self.k):
            px += self.pi[k] * self._gaussian(X, k)
        return np.log(px + 1e-300).sum()

    def fit(self, X, km=None):
        self.init_params(X, km=km)
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
                print(f"  GMM k={self.k}: convergió en iteración {i+1}")
                break
        
        return self

    def predict(self, X):
        """Asignación dura: argmax de responsabilidades."""
        r = self.e_step(X)
        return r.argmax(axis=1)