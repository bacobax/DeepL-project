import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from utils.datasets import CLASS_NAMES
import torch


def cluster_categories(clip_model, base_classes, train_base, device):
    class_feature = {}
    with torch.no_grad():
        for c in base_classes:
            imgs_c = [img for img, label in train_base if label == c]
            features = [clip_model.encode_image(img.unsqueeze(0).to(device)).cpu().numpy() for img in imgs_c]
            class_feature[c] = np.mean(features, axis=0)

    #class_ft_array = np.array([class_feature[c][0] for c in base_classes])

    cat2idx = {}
    idx2cat = {}
    class_ft_array = []
    for i, c in enumerate(base_classes):
        cat2idx[c] = i
        idx2cat[i] = c
        class_ft_array.append(class_feature[c][0])

    pca = PCA(n_components=0.95)  # Keep 95% of variance
    X_reduced = pca.fit_transform(class_ft_array)

    n_clusters = 2  # or more if needed

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_reduced)  # shape: [n_base_classes x 1]

    cluster_labels = {
        idx2cat[i]: cluster for i, cluster in enumerate(cluster_labels)
    }

    cluster_labels_text = {
        CLASS_NAMES[base_class]: int(cluster) for base_class, cluster in enumerate(cluster_labels)
    }

    return cluster_labels, cluster_labels_text

