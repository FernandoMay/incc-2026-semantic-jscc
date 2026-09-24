"""SD-02: semantic JSCC prototype over Rayleigh fading."""

from pathlib import Path
import json
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestCentroid
from sklearn.preprocessing import StandardScaler


def make_data(seed=20260915, samples=1200, features=64, classes=4):
    rng = np.random.default_rng(seed)
    labels = np.arange(samples) % classes
    centers = rng.normal(0, 2.0, (classes, features))
    x = centers[labels] + rng.normal(0, 0.8, (samples, features))
    return x, labels


def rayleigh_channel(x, snr_db, rng):
    h = rng.rayleigh(1.0, size=x.shape)
    signal_power = np.mean((h * x) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, np.sqrt(noise_power), size=x.shape)
    return h * x + noise, h


def nearest_centroid_accuracy(train_x, train_y, test_x, test_y):
    classifier = NearestCentroid().fit(train_x, train_y)
    return float(classifier.score(test_x, test_y))


def evaluate(seed=20260915, snr_values=(0, 2, 5, 10), latent_dim=8):
    x, y = make_data(seed=seed)
    scaler = StandardScaler().fit(x)
    x = scaler.transform(x)
    train_x, test_x, train_y, test_y = train_test_split(
        x, y, test_size=0.3, random_state=seed, stratify=y
    )
    pca = PCA(n_components=latent_dim, random_state=seed).fit(train_x)
    train_latent = pca.transform(train_x)
    test_latent = pca.transform(test_x)
    rng = np.random.default_rng(seed)
    output = {}
    for snr in snr_values:
        semantic_channel, _ = rayleigh_channel(test_latent, snr, rng)
        raw_channel, _ = rayleigh_channel(test_x, snr, rng)
        semantic_reconstruction = pca.inverse_transform(semantic_channel)
        raw_reconstruction = raw_channel
        output[str(snr)] = {
            "semantic_accuracy": nearest_centroid_accuracy(
                train_x, train_y, semantic_reconstruction, test_y
            ),
            "raw_accuracy": nearest_centroid_accuracy(
                train_x, train_y, raw_reconstruction, test_y
            ),
            "semantic_dimensions": latent_dim,
            "raw_dimensions": x.shape[1],
            "bandwidth_ratio": latent_dim / x.shape[1],
        }
    return {
        "seed": seed,
        "samples": len(x),
        "features": x.shape[1],
        "latent_dim": latent_dim,
        "snr": output,
    }


if __name__ == "__main__":
    result = evaluate()
    Path("data").mkdir(exist_ok=True)
    Path("data/metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
