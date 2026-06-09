import numpy as np
import matplotlib.pyplot as plt

def silhouette_score(X, labels):
    N = len(X)
    clusters = np.unique(labels)
    s = np.zeros(N)

    for i in range(N):
        own = labels[i]
        own_pts = X[labels == own]

        # a(i): distancia promedio a puntos del mismo cluster
        if len(own_pts) > 1:
            a = np.sqrt(((X[i] - own_pts) ** 2).sum(axis=1)).sum() / (len(own_pts) - 1)
        else:
            a = 0.0

        # b(i): distancia promedio al cluster vecino más cercano
        b = np.inf
        for c in clusters:
            if c == own:
                continue
            other_pts = X[labels == c]
            mean_dist = np.sqrt(((X[i] - other_pts) ** 2).sum(axis=1)).mean()
            if mean_dist < b:
                b = mean_dist

        s[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0

    return s.mean()

def silhouette_score_sampled(X, labels, n=500, seed=42):
    idx = np.random.default_rng(seed).choice(len(X), size=n, replace=False)
    return silhouette_score(X[idx], labels[idx])

def compute_clustering_metrics(K_RANGE, results_km, results_gmm, X_sample):
    ks = list(K_RANGE)
    
    inertias  = [results_km[k].inertia                                          for k in K_RANGE]
    sil_km    = [silhouette_score_sampled(X_sample, results_km[k].labels)       for k in K_RANGE]
    ll        = [results_gmm[k].log_likelihoods[-1]                             for k in K_RANGE]
    sil_gmm   = [silhouette_score_sampled(X_sample, results_gmm[k].predict(X_sample)) for k in K_RANGE]

    return {
        "ks":             ks,
        "inertias":       inertias,
        "marginal_i":     list(np.diff(inertias)),
        "sil_km":         sil_km,
        "ll":             ll,
        "marginal_ll":    list(np.diff(ll)),
        "sil_gmm":        sil_gmm,
    }

def compute_pairwise_distances(X):
    """Distancias euclidianas al cuadrado entre todos los pares. (N, N)"""
    sum_sq = (X ** 2).sum(axis=1)
    D = sum_sq[:, None] + sum_sq[None, :] - 2 * (X @ X.T)
    return np.maximum(D, 0)   # evitar negativos por floating point


def gaussian_perplexity(D_row, perplexity, n_iter=50, tol=1e-5):
    """
    Busca el sigma para un punto tal que la perplejidad de su distribución
    sea igual a la deseada. Usa búsqueda binaria sobre beta = 1/(2*sigma²).
    Devuelve p_i|j (distribución condicional para ese punto).
    """
    target_entropy = np.log(perplexity)
    beta_min, beta_max = -np.inf, np.inf
    beta = 1.0

    for _ in range(n_iter):
        exp_D   = np.exp(-D_row * beta)
        sum_exp = exp_D.sum()
        if sum_exp == 0:
            sum_exp = 1e-10
        p       = exp_D / sum_exp
        entropy = -np.sum(p * np.log(p + 1e-10))

        if abs(entropy - target_entropy) < tol:
            break

        if entropy > target_entropy:   # sigma muy grande → achicar
            beta_min = beta
            beta = beta * 2 if beta_max == np.inf else (beta + beta_max) / 2
        else:                          # sigma muy chico → agrandar
            beta_max = beta
            beta = beta / 2 if beta_min == -np.inf else (beta + beta_min) / 2

    return p


def compute_P(X, perplexity=30):
    """
    Matriz de afinidades simétricas P en el espacio original. (N, N)
    P_ij = (p_j|i + p_i|j) / (2N)
    """
    N  = X.shape[0]
    D  = compute_pairwise_distances(X)
    P  = np.zeros((N, N))

    for i in range(N):
        d_row    = D[i].copy()
        d_row[i] = np.inf                          # excluir distancia a sí mismo
        P[i]     = gaussian_perplexity(d_row, perplexity)
        P[i, i]  = 0.0

    P = (P + P.T) / (2 * N)
    P = np.maximum(P, 1e-12)                       # evitar log(0)
    return P


def tsne(X, n_components=2, perplexity=30, n_iter=1000,
         lr=200, momentum=0.9, seed=42):
    """
    t-SNE desde cero.
    - X: (N, D) ya reducido (espacio latente PCA o AE)
    - Devuelve Y: (N, n_components)
    """
    N = X.shape[0]
    rng = np.random.default_rng(seed)

    # ── 1. Afinidades en espacio original ────────────────────
    print("Calculando matriz P...")
    P = compute_P(X, perplexity)
    P *= 4                                         # early exaggeration

    # ── 2. Inicialización aleatoria del embedding ─────────────
    Y      = rng.normal(0, 1e-4, (N, n_components))
    Y_prev = Y.copy()
    gains  = np.ones_like(Y)                       # gains adaptivos por dimensión

    for t in range(1, n_iter + 1):

        # afinidades Q en espacio reducido (distribución t de Student)
        D_Y    = compute_pairwise_distances(Y)
        Q_num  = 1 / (1 + D_Y)
        np.fill_diagonal(Q_num, 0)
        Q      = Q_num / Q_num.sum()
        Q      = np.maximum(Q, 1e-12)

        # gradiente: dC/dY_i
        PQ   = P - Q                               # (N, N)
        grad = np.zeros_like(Y)
        for i in range(N):
            grad[i] = 4 * ((PQ[:, i] * Q_num[:, i])[:, None] * (Y[i] - Y)).sum(axis=0)

        # gains adaptivos (acelera si el gradiente mantiene signo)
        gains = (gains + 0.2) * ((grad > 0) != (Y - Y_prev > 0)) + \
                (gains * 0.8) * ((grad > 0) == (Y - Y_prev > 0))
        gains = np.maximum(gains, 0.01)

        # actualización con momentum
        step   = momentum * (Y - Y_prev) - lr * gains * grad
        Y_prev = Y.copy()
        Y      = Y + step
        Y     -= Y.mean(axis=0)                    # centrar

        # quitar early exaggeration después de 250 iteraciones
        if t == 250:
            P /= 4

        if t % 100 == 0:
            kl = (P * np.log(P / Q)).sum()
            print(f"  iter {t:4d}  KL={kl:.4f}")

    return Y

def plot_tsne(ax, Y, labels, title, is_cluster=True):
    LABEL_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
    }
    for c in np.unique(labels):
        mask = labels == c
        label = f'Cluster {c}' if is_cluster else LABEL_NAMES[c]
        ax.scatter(Y[mask, 0], Y[mask, 1], s=5, alpha=0.6, label=label)
    ax.set_title(title)
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3, loc='best', fontsize=8)


def plot_tsne_clusters(Y_pca, Y_ae, labels_km_pca, labels_km_ae,
                       labels_gmm_pca, labels_gmm_ae, k):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    plot_tsne(axes[0,0], Y_pca, labels_km_pca,  f'KMeans k={k} — espacio PCA')
    plot_tsne(axes[0,1], Y_ae,  labels_km_ae,   f'KMeans k={k} — espacio AE')
    plot_tsne(axes[1,0], Y_pca, labels_gmm_pca, f'GMM k={k} — espacio PCA')
    plot_tsne(axes[1,1], Y_ae,  labels_gmm_ae,  f'GMM k={k} — espacio AE')
    plt.suptitle('t-SNE: clusters en espacio latente', fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_tsne_true_labels(Y_pca, Y_ae, y_sample):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plot_tsne(axes[0], Y_pca, y_sample, 'Clases reales — espacio PCA', is_cluster=False)
    plot_tsne(axes[1], Y_ae,  y_sample, 'Clases reales — espacio AE',  is_cluster=False)
    plt.suptitle('t-SNE: clases reales (referencia)', fontsize=14)
    plt.tight_layout()
    plt.show()