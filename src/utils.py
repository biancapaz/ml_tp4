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

def show_images(X, y, n=15, indexes=None, titles=None, random_state=42):
    rng = np.random.default_rng(random_state)
    idx = indexes if indexes is not None else rng.choice(len(X), n, replace=False)

    fig, axes = plt.subplots(1, len(idx), figsize=(2 * len(idx), 2.5))
    for ax, i in zip(axes, idx):
        img = X[i].reshape(28, 28)

        ax.imshow(img, cmap='gray')
        label = titles[i] if titles is not None else LABEL_NAMES.get(y[i], y[i])
        ax.set_title(label, fontsize=8)
        ax.axis('off')

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

def find_k_90(pca, threshold=0.9):
    var_ratios = pca.explained_variance_ratio()
    cumulative = np.cumsum(var_ratios)
    k = np.argmax(cumulative >= threshold) + 1
    print(f"Componentes para explicar {threshold*100:.0f}% de varianza: {k}")
    return k

def plot_explained_variance(pca, threshold=0.9):
    cumvar = np.cumsum(pca.explained_variance_ratio())
    k = find_k_90(pca, threshold=threshold)

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

def reconstruct(X_std, pca, k):
    pc = pca.components[:, :k]
    X_proj = pca.transform_pca(X_std, k=k)
    X_recon = np.dot(X_proj, pc.T)

    std = np.where(np.isclose(pca.std, 0), 1, pca.std)
    X_recon = (X_recon * std) + pca.mean

    # clampear al rango válido
    X_recon = np.clip(X_recon, 0, 1)

    return X_recon

def plot_reconstruction(X_original, X_reconstructed, indices):
    n = len(indices)
    fig, axes = plt.subplots(2, n, figsize=(2 * n, 4))

    for col, i in enumerate(indices):
        for row, (imgs, title) in enumerate([(X_original, "Original"), (X_reconstructed, "PCA recon."),]):
            ax = axes[row, col]
            ax.imshow(imgs[i].reshape(28, 28), cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(title, fontsize=8)

    fig.suptitle(f"Reconstrucción PCA (k={X_reconstructed.shape[1] if hasattr(X_reconstructed, 'shape') else '?'})")
    plt.tight_layout()
    plt.show()