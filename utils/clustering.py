import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
from utils.datasets import get_data, base_novel_categories, split_data, CLASS_NAMES
import clip
import os
import pickle


def cluster_categories(
    clip_model, base_classes, train_base, device, n_clusters=2, variance=0.95
):
    class_feature = {}
    with torch.no_grad():
        for c in tqdm(base_classes, desc="Processing classes"):
            imgs_c = [img for img, label in train_base if label == c]
            features = [
                clip_model.encode_image(img.unsqueeze(0).to(device)).cpu().numpy()
                for img in imgs_c
            ]
            class_feature[c] = np.mean(features, axis=0)

    # class_ft_array = np.array([class_feature[c][0] for c in base_classes])

    cat2idx = {}
    idx2cat = {}
    class_ft_array = []
    for i, c in enumerate(base_classes):
        cat2idx[c] = i
        idx2cat[i] = c
        class_ft_array.append(class_feature[c][0])

    pca = PCA(n_components=variance)
    X_reduced = pca.fit_transform(class_ft_array)

    print(f"Reduced feature shape: {X_reduced.shape}, Variance explained: {pca.explained_variance_ratio_.sum()}")

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(X_reduced)  # shape: [n_base_classes x 1]

    cluster_labels = {idx2cat[i]: cluster for i, cluster in enumerate(cluster_labels)}

    cluster_labels_text = {
        CLASS_NAMES[base_class]: int(cluster)
        for base_class, cluster in enumerate(cluster_labels)
    }

    return cluster_labels, cluster_labels_text


def conditional_clustering(n_cluster, variance, cnn_safe, c_m, classes, dataset, device):
    save_dir = f"clustering_split/cluster_labels_{n_cluster}_{variance}_{cnn_safe}"
    os.makedirs(save_dir, exist_ok=True)

    int_categories_path = os.path.join(save_dir, "int_categories.pkl")
    text_categories_path = os.path.join(save_dir, "text_categories.pkl")

    if os.path.exists(int_categories_path) and os.path.exists(text_categories_path):
        print("Loading existing cluster labels...")
        with open(int_categories_path, "rb") as f:
            cluster_labels = pickle.load(f)
            cluster_dict_int = {int(k): v for k, v in cluster_labels.items()}
        with open(text_categories_path, "rb") as f:
            cluster_labels_text = pickle.load(f)

    else:
        print("Clustering base classes...")
        # cluster the base classes
        cluster_labels, cluster_labels_text = cluster_categories(
            c_m, classes, dataset, device, n_clusters=n_cluster, variance=variance
        )

        with open(int_categories_path, "wb") as f:
            pickle.dump(cluster_labels, f)
        with open(text_categories_path, "wb") as f:
            pickle.dump(cluster_labels_text, f)

    return cluster_dict_int, cluster_labels_text


if __name__ == "__main__":

    N_CLUSTERS = 2
    VARIANCE = 0.95
    CNN = "ViT-B/32"

    # set device to either cuda, mps or cpu

    DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    CNN_SAFE = CNN.replace("/", "_")
    # initialize clip model with ViT
    clip_model, preprocess = clip.load(CNN)
    clip_model = clip_model.to(DEVICE)

    train_set, _, _ = get_data(data_dir="../data", transform=preprocess)

    # split classes into base and novel
    base_classes, _ = base_novel_categories(train_set)

    # split the three datasets
    train_base, _ = split_data(train_set, base_classes)

    cls_cluster_dict_int, cluster_labels_text = conditional_clustering(
        N_CLUSTERS, VARIANCE, CNN_SAFE, clip_model, base_classes, train_base, DEVICE
    )
    print(cls_cluster_dict_int, cluster_labels_text)
