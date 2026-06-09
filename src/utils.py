import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Mapeo canónico de Fashion MNIST
LABEL_NAMES = {
    0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress", 4: "Coat",
    5: "Sandal", 6: "Shirt", 7: "Sneaker", 8: "Bag", 9: "Ankle boot"
}

def dataset_summary(X, y):
    n_samples, n_features = X.shape
    classes, counts = np.unique(y, return_counts=True)

    summary_data = pd.DataFrame({
        "Métrica": ["Samples", "Features", "Classes", "Pixel range", "Dtype", "NaN values", "Balance"],
        "Valor": [
            f"{n_samples:,}",
            f"{n_features} ({int(n_features**0.5)}x{int(n_features**0.5)} px)",
            len(classes),
            f"[{X.min()}, {X.max()}]",
            X.dtype,
            np.isnan(X.astype(float)).sum(),
            "uniform" if counts.std() == 0 else f"std={counts.std():.1f}",
        ]
    })

    summary_labels = pd.DataFrame({
        "Class ID": classes,
        "Name": [LABEL_NAMES[c] for c in classes],
        "Count": counts,
        "Pct (%)": (counts / n_samples * 100).round(2),
    })

    return summary_data, summary_labels

def analyze_pixels(X, y):
    """Calcula stats por clase y genera todos los plots derivados."""
    stats = []
    for cls in sorted(np.unique(y)):
        subset = X[y == cls].astype(float)
        stats.append({
            "class":       LABEL_NAMES[cls],
            "mean_px":     subset.mean().round(2),
            "std_px":      subset.std().round(2)
        })

    df = pd.DataFrame(stats)
    plot_pixel_stats(df)
    return df

def plot_pixel_stats(df):
    """Visualiza media y std de intensidad por clase (barras horizontales)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, col, title, color in zip(
        axes,
        ["mean_px", "std_px"],
        ["Intensidad media de píxeles por clase", "Std de intensidad por clase"],
        ["steelblue", "coral"]
    ):
        bars = ax.barh(df["class"], df[col], color=color, alpha=0.8)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Intensidad (0.0–1.0)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(5))

    plt.tight_layout()
    plt.show()

def show_images(X, y, n=15, indices=None, axes=None, row_title=None, random_state=42):
    X = np.array(X)
    rng = np.random.default_rng(random_state)
    idx = indices if indices is not None else rng.choice(len(X), n, replace=False)

    # si no recibe axes externos, crea su propia figura (uso standalone)
    standalone = axes is None
    if standalone:
        fig, axes = plt.subplots(1, len(idx), figsize=(2 * len(idx), 2.5))

    for col, i in enumerate(idx):
        axes[col].imshow(X[i].reshape(28, 28), cmap="gray")
        axes[col].set_title(LABEL_NAMES.get(y[i], y[i]), fontsize=8)
        axes[col].axis("off")

    if row_title:
        axes[0].set_ylabel(row_title, fontsize=9)

    if standalone:
        plt.tight_layout()
        plt.show()

    return idx

def show_separate_classes(X, y, classes, n_per_class, random_state = 42):
    rng = np.random.default_rng(random_state)
    n_cls = len(classes)
    fig, axes = plt.subplots(n_cls, n_per_class, figsize=(2 * n_per_class, 2 * n_cls))

    for row_idx, cls in enumerate(classes):
        cls_idxs = np.where(y == cls)[0]
        selected = rng.choice(cls_idxs, n_per_class, replace=False)

        for col_idx, i in enumerate(selected):
            ax = axes[row_idx, col_idx]
            img = X[i].reshape(28, 28)
            ax.imshow(img, cmap="gray")
            ax.axis("off")
        
        axes[row_idx, 0].set_ylabel(LABEL_NAMES.get(cls, cls), fontsize=8, rotation=0, labelpad=55, va="center")
    
    fig.suptitle("Ejemplos por clase", y=1.01)
    plt.tight_layout()
    plt.show()

def show_proportions(df, title=None):
    """Gráfico de barras horizontales — más legible que pie para comparar magnitudes."""
    counts = df["label"].map(LABEL_NAMES).value_counts().sort_values()

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(counts.index, counts.values,
                   color=plt.get_cmap("tab10").colors[:len(counts)])
    ax.bar_label(bars, fmt="%d", padding=3, fontsize=8)
    ax.set_xlabel("Cantidad de muestras")
    title = title if title is not None else "Distribución de clases — Fashion MNIST"
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_explained_variance(pca, threshold=0.9):
    evr = pca.explained_variance_ratio()
    cumvar = np.cumsum(evr)
    
    k = np.argmax(cumvar >= threshold) + 1

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=1.5, color="steelblue")
    ax.axhline(threshold, color="coral", linestyle="--", linewidth=1, label=f"{threshold*100:.0f}% varianza")
    ax.axvline(k, color="darkgreen", linestyle="--", linewidth=1, label=f"k={k}")
    ax.scatter([k], [cumvar[k - 1]], color="darkgreen", zorder=5)
    ax.set_xlabel("Número de componentes")
    ax.set_ylabel("Varianza explicada acumulada")
    ax.set_title("Varianza explicada acumulada — PCA")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

    return k

def plot_reconstruction(X_original, X_reconstructed, y, n=10, seed=42):
    indices = np.random.default_rng(seed).choice(len(X_original), n, replace=False)
    fig, axes = plt.subplots(2, len(indices), figsize=(2 * len(indices), 4))
    
    show_images(X_original, y, indices=indices, axes=axes[0], row_title="Original")
    show_images(X_reconstructed, y, indices=indices, axes=axes[1], row_title="Reconstruida")
    
    plt.tight_layout()
    plt.show()

def plot_loss(train_losses, test_losses):
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(train_losses, label="Train")
    ax.plot(test_losses,  label="Test")
    ax.set_xlabel("Época")
    ax.set_ylabel("MSE")
    ax.set_title("Curva de entrenamiento — Autoencoder")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()

def plot_clustering_metrics(metrics_pca, metrics_ae):
    ks     = metrics_pca["ks"]
    ks_mid = ks[1:]   # para las ganancias marginales (diff achica en 1)

    def plot_two(ax, ks, y_pca, y_ae, title, ylabel):
        ax.plot(ks, y_pca, marker='o', label='PCA', color='orange')
        ax.plot(ks, y_ae,  marker='o', label='AE',  color='steelblue')
        ax.set_title(title); ax.set_xlabel('K'); ax.set_ylabel(ylabel)
        ax.legend(); ax.grid(True, alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))

    plot_two(axes[0,0], ks,     metrics_pca["inertias"],    metrics_ae["inertias"],    'KMeans — Inercia vs K',      'Inercia')
    plot_two(axes[0,1], ks_mid, metrics_pca["marginal_i"],  metrics_ae["marginal_i"],  'KMeans — Ganancia marginal', 'Δ Inercia')
    plot_two(axes[0,2], ks,     metrics_pca["sil_km"],      metrics_ae["sil_km"],      'KMeans — Silhouette vs K',   'Silhouette')
    plot_two(axes[1,0], ks,     metrics_pca["ll"],          metrics_ae["ll"],          'GMM — Log-likelihood vs K',  'Log-likelihood')
    plot_two(axes[1,1], ks_mid, metrics_pca["marginal_ll"], metrics_ae["marginal_ll"], 'GMM — Ganancia marginal',    'Δ Log-likelihood')
    plot_two(axes[1,2], ks,     metrics_pca["sil_gmm"],     metrics_ae["sil_gmm"],     'GMM — Silhouette vs K',      'Silhouette')

    plt.suptitle('Clustering: PCA vs AE latent space', fontsize=14, y=1.01)
    plt.tight_layout()
    plt.show()