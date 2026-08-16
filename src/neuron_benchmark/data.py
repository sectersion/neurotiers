import gzip
import os
import struct
import urllib.request
from pathlib import Path

import torch


def _fetch_mnist(url, filename):
    data_dir = Path.home() / ".cache" / "neuron_benchmark"
    data_dir.mkdir(parents=True, exist_ok=True)
    filepath = data_dir / filename
    if not filepath.exists():
        urllib.request.urlretrieve(url, filepath)
    return filepath


def _load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = f.read()
        images = torch.frombuffer(data, dtype=torch.uint8).reshape(num, rows, cols)
    return images


def _load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = torch.frombuffer(f.read(), dtype=torch.uint8).long()
    return labels


def _download_mnist(train=True):
    base = "http://yann.lecun.com/exdb/mnist"
    if train:
        images_url = f"{base}/train-images-idx3-ubyte.gz"
        labels_url = f"{base}/train-labels-idx1-ubyte.gz"
        images_file = "train-images-idx3-ubyte.gz"
        labels_file = "train-labels-idx1-ubyte.gz"
    else:
        images_url = f"{base}/t10k-images-idx3-ubyte.gz"
        labels_url = f"{base}/t10k-labels-idx1-ubyte.gz"
        images_file = "t10k-images-idx3-ubyte.gz"
        labels_file = "t10k-labels-idx1-ubyte.gz"
    images_path = _fetch_mnist(images_url, images_file)
    labels_path = _fetch_mnist(labels_url, labels_file)
    return _load_images(images_path), _load_labels(labels_path)


def _download_fashion_mnist(train=True):
    base = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com"
    if train:
        images_url = f"{base}/train-images-idx3-ubyte.gz"
        labels_url = f"{base}/train-labels-idx1-ubyte.gz"
        images_file = "fashion-train-images-idx3-ubyte.gz"
        labels_file = "fashion-train-labels-idx1-ubyte.gz"
    else:
        images_url = f"{base}/t10k-images-idx3-ubyte.gz"
        labels_url = f"{base}/t10k-labels-idx1-ubyte.gz"
        images_file = "fashion-t10k-images-idx3-ubyte.gz"
        labels_file = "fashion-t10k-labels-idx1-ubyte.gz"
    images_path = _fetch_mnist(images_url, images_file)
    labels_path = _fetch_mnist(labels_url, labels_file)
    return _load_images(images_path), _load_labels(labels_path)


def _try_torchvision_mnist(train=True):
    try:
        from torchvision import datasets
        dataset = datasets.MNIST(root=Path.home() / ".cache" / "torch", train=train, download=True)
        images = dataset.data.float() / 255.0
        labels = dataset.targets
        return images, labels
    except Exception:
        return None, None


def _try_torchvision_fashion_mnist(train=True):
    try:
        from torchvision import datasets
        dataset = datasets.FashionMNIST(root=Path.home() / ".cache" / "torch", train=train, download=True)
        images = dataset.data.float() / 255.0
        labels = dataset.targets
        return images, labels
    except Exception:
        return None, None


def load_mnist(train=True, samples=None, scale=10.0, seed=0):
    """Load MNIST dataset as Poisson spike-train rates.

    Args:
        train: If True, load training set; otherwise test set.
        samples: Limit number of samples (None for full dataset).
        scale: Multiplier for pixel values to set spike generation rate.
               Rate = pixel_value * scale, clamped to [0, 1].
        seed: Random seed for sampling.

    Returns:
        (x, y) where x is (N, 784) rate tensor and y is (N,) labels.
    """
    images, labels = _try_torchvision_mnist(train)
    if images is None:
        images, labels = _download_mnist(train)

    images = images.float() / 255.0
    images = images.flatten(start_dim=1)

    generator = torch.Generator().manual_seed(seed)
    if samples is not None and samples < len(images):
        indices = torch.randperm(len(images), generator=generator)[:samples]
        images = images[indices]
        labels = labels[indices]

    rates = (images * scale).clamp(0, 1)
    return rates, labels


def load_fashion_mnist(train=True, samples=None, scale=10.0, seed=0):
    """Load Fashion-MNIST dataset as Poisson spike-train rates.

    Args:
        train: If True, load training set; otherwise test set.
        samples: Limit number of samples (None for full dataset).
        scale: Multiplier for pixel values to set spike generation rate.
               Rate = pixel_value * scale, clamped to [0, 1].
        seed: Random seed for sampling.

    Returns:
        (x, y) where x is (N, 784) rate tensor and y is (N,) labels.
    """
    images, labels = _try_torchvision_fashion_mnist(train)
    if images is None:
        images, labels = _download_fashion_mnist(train)

    images = images.float() / 255.0
    images = images.flatten(start_dim=1)

    generator = torch.Generator().manual_seed(seed)
    if samples is not None and samples < len(images):
        indices = torch.randperm(len(images), generator=generator)[:samples]
        images = images[indices]
        labels = labels[indices]

    rates = (images * scale).clamp(0, 1)
    return rates, labels


def synthetic_data(samples=128, features=8, classes=2, seed=0):
    """Create separable event rates; no network or dataset download required."""
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(classes, (samples,), generator=generator)
    centers = torch.linspace(0.2, 0.8, classes).unsqueeze(1)
    rates = centers[labels] + 0.12 * torch.randn(samples, features, generator=generator)
    return rates.clamp(0, 1), labels
