import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm
import torch
from datasets import get_data, base_novel_categories, split_data, CLASS_NAMES
import clip
import os
import pickle
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from collections import Counter, deque


import random
from collections import deque

def create_cluster_from_ordered_list(ordered_categories, split_ratio):
    """
    Converts a list of categories into a cluster dict using split_ratio.
    """
    n = len(ordered_categories)
    n_zeros = int(n * split_ratio)
    return {
        cat: 0 if i < n_zeros else 1
        for i, cat in enumerate(ordered_categories)
    }

def rotating_cluster_generator_shift(categories, split_ratio, steps=1, seed=None):
    """
    Yields clusters by cyclically rotating the category list.

    Args:
        categories (list): List of category identifiers.
        split_ratio (float): Ratio of cluster 0 elements.
        seed (int, optional): Random seed for reproducibility.

    Yields:
        dict: Cluster mapping.
    """
    cat_list = list(categories)
    if seed is not None:
        random.seed(seed)
    random.shuffle(cat_list)
    cat_deque = deque(cat_list)

    while True:
        cluster = create_cluster_from_ordered_list(cat_deque, split_ratio)
        cat0 = [cat for cat, c in cluster.items() if c == 0]
        cat1 = [cat for cat, c in cluster.items() if c == 1]
        yield cluster, cat0, cat1
        cat_deque.rotate(-steps)  # rotate left

def cluster_categories(device, cnn, n_clusters=2, variance=0.95, data_dir="../data"):
    """
    Clusters base classes using visual features extracted from a CLIP model. Applies PCA to reduce dimensionality
    and Agglomerative Clustering on cosine distances to group the categories.

    Args:
        device (torch.device): The device to run computations on (CPU/GPU).
        cnn (str): CLIP model architecture name (e.g., "ViT-B/32").
        n_clusters (int): Number of clusters to generate.
        variance (float): Variance ratio to preserve during PCA.

    Returns:
        Tuple[Dict[int, int], Dict[str, int]]: Two dictionaries mapping class indices and class names to cluster IDs.
    """

    # initialize clip model with ViT
    clip_model, preprocess = clip.load(cnn)
    clip_model = clip_model.to(device)

    train_set, _, _ = get_data(data_dir=data_dir, transform=preprocess)

    # split classes into base and novel
    base_classes, _ = base_novel_categories(train_set)

    # split the three datasets
    train_base, _ = split_data(train_set, base_classes)

    class_feature = {}
    with torch.no_grad():
        for c in tqdm(base_classes, desc="Processing classes"):
            imgs_c = []
            # Create a DataLoader to iterate over the dataset properly
            dataloader = torch.utils.data.DataLoader(train_base, batch_size=1, shuffle=False)
            for img, label in dataloader:
                if label.item() == c:
                    imgs_c.append(img.squeeze(0))
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

    print(
        f"Reduced feature shape: {X_reduced.shape}, Variance explained: {pca.explained_variance_ratio_.sum()}"
    )

    cosine_dist = cosine_distances(X_reduced)
    # Step 5: Agglomerative clustering
    agglo = AgglomerativeClustering(
        n_clusters=n_clusters, metric="precomputed", linkage="average"
    )
    cluster_labels = agglo.fit_predict(cosine_dist)

    cluster_labels = {idx2cat[i]: cluster for i, cluster in enumerate(cluster_labels)}

    cluster_labels_text = {
        CLASS_NAMES[base_class]: int(cluster)
        for base_class, cluster in enumerate(cluster_labels)
    }

    torch.cuda.empty_cache()

    return cluster_labels, cluster_labels_text


def random_clustering(
    n_cluster,
    seed=42,
    data_dir="../data",
    distribution="uniform",
    split_ratio=0.7,  # Only for bipartite
):
    """
    Generates random cluster assignments for a given number of clusters.

    Args:
        n_cluster (int): Number of clusters to generate.
        seed (int): Random seed for reproducibility.
        data_dir (str): Directory where the dataset is stored.
        distribution (str): Distribution type for cluster assignment. Options are "uniform", "random", "sequential", or "bipartite".
        split_ratio (float): Percentage of classes in the larger cluster (only used for "bipartite").

    Returns:
        Tuple[
            Dict[int, int],        # class_id -> cluster_id
            Dict[str, int],        # class_name -> cluster_id
            List[int],             # class_ids in cluster 0
            List[int],             # class_ids in cluster 1
        ]
    """

    np.random.seed(seed)
    train_set, _, _ = get_data(data_dir=data_dir)
    base_classes, _ = base_novel_categories(train_set)

    cluster_labels = {}

    if distribution == "uniform":
        shuffled = np.random.permutation(base_classes)
        for i, cls in enumerate(shuffled):
            cluster_id = i % n_cluster
            cluster_labels[cls] = cluster_id

    elif distribution == "random":
        for cls in base_classes:
            cluster_id = np.random.choice(range(n_cluster))
            cluster_labels[cls] = cluster_id

    elif distribution == "sequential":
        for i, cls in enumerate(base_classes):
            cluster_id = i % n_cluster
            cluster_labels[cls] = cluster_id

    cluster_labels_text = {
        CLASS_NAMES[cls]: int(cluster_labels[cls])
        for cls in cluster_labels
    }

    cluster_dict_int = {int(k): v for k, v in cluster_labels.items()}

    return cluster_dict_int, cluster_labels_text


def conditional_clustering(n_cluster, variance, cnn, device, data_dir="../data"):
    """
    Loads existing cluster labels from disk if available, otherwise computes and saves new cluster assignments.

    Args:
        n_cluster (int): Number of clusters to generate.
        variance (float): Variance ratio to preserve during PCA.
        cnn (str): CLIP model architecture name (used for naming output files).
        device (torch.device): The device to run computations on.

    Returns:
        Tuple[Dict[int, int], Dict[str, int]]: Dictionaries for integer-labeled and text-labeled cluster assignments.
    """
    cnn_sanitized = cnn.replace("/", "_")
    save_dir = f"clustering_split/cluster_labels_{n_cluster}_{variance}_{cnn_sanitized}"
    os.makedirs(save_dir, exist_ok=True)

    int_categories_path = os.path.join(save_dir, "int_categories.pkl")
    text_categories_path = os.path.join(save_dir, "text_categories.pkl")

    if os.path.exists(int_categories_path) and os.path.exists(text_categories_path):
        print("🟩 CLUSTERS FILES FOUND. Loading existing cluster labels...")
        with open(int_categories_path, "rb") as f:
            cluster_labels = pickle.load(f)
            cluster_dict_int = {int(k): v for k, v in cluster_labels.items()}
        with open(text_categories_path, "rb") as f:
            cluster_labels_text = pickle.load(f)

    else:
        print("🟧 NO CLUSTERS FILES FOUND. Loading existing cluster labels...")
        # cluster the base classes
        cluster_labels, cluster_labels_text = cluster_categories(
            device, n_clusters=n_cluster, variance=variance, cnn=cnn, data_dir=data_dir
        )
        cluster_dict_int = {int(k): v for k, v in cluster_labels.items()}
        with open(int_categories_path, "wb") as f:
            pickle.dump(cluster_labels, f)
        with open(text_categories_path, "wb") as f:
            pickle.dump(cluster_labels_text, f)
    # Count samples in each cluster
    cluster_counts = Counter(cluster_dict_int.values())
    for cluster_id in range(n_cluster):
        print(f"Cluster {cluster_id} count: {cluster_counts.get(cluster_id, 0)}")

    return cluster_dict_int, cluster_labels_text

def rotate_clusters(split_ratio: float, data_dir="../data"):
    """
    Creates a valid cluster configuration where split_ratio*N categories belong to cluster 0,
    and the rest belong to cluster 1.
    
    Args:
        split_ratio (float): Ratio of categories that should belong to cluster 0 (0 < split_ratio < 1)
        data_dir (str): Directory where the dataset is stored
        
    Returns:
        dict: Mapping from category index to cluster (0 or 1)
    """
    # Get the dataset to determine number of categories
    train_set, _, _ = get_data(data_dir=data_dir)
    base_classes, _ = base_novel_categories(train_set)
    N = len(base_classes)
    
    # Calculate how many categories should be in cluster 0
    n_zeros = int(N * split_ratio)
    n_ones = N - n_zeros
    
    # Create the cluster assignment
    cluster_assignments = {}
    for i, category in enumerate(base_classes):
        if i < n_zeros:
            cluster_assignments[category] = 0
        else:
            cluster_assignments[category] = 1
    
    return cluster_assignments

def clusters_history(split_ratio: float, data_dir="../data"):
    """
    Creates an array of all possible cluster configurations as heterogeneously as possible.
    Each configuration follows the split_ratio constraint.
    
    Args:
        split_ratio (float): Ratio of categories that should belong to cluster 0
        data_dir (str): Directory where the dataset is stored
        
    Returns:
        list: List of cluster configurations (each is a dict mapping category to cluster)
    """
    # Get the dataset to determine number of categories
    train_set, _, _ = get_data(data_dir=data_dir)
    base_classes, _ = base_novel_categories(train_set)
    N = len(base_classes)
    
    # Calculate how many categories should be in cluster 0
    n_zeros = int(N * split_ratio)
    n_ones = N - n_zeros
    
    # Generate all possible combinations of n_zeros categories for cluster 0
    from itertools import combinations
    
    all_clusters = []
    for zero_indices in combinations(range(N), n_zeros):
        cluster_config = {}
        for i, category in enumerate(base_classes):
            if i in zero_indices:
                cluster_config[category] = 0
            else:
                cluster_config[category] = 1
        all_clusters.append(cluster_config)
    
    return all_clusters


# if __name__ == "__main__":

#     N_CLUSTERS = 2
#     VARIANCE = 0.95
#     CNN = "ViT-B/32"

#     # set device to either cuda, mps or cpu

#     DEVICE = torch.device(
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps" if torch.backends.mps.is_available() else "cpu"
#     )

#     cls_cluster_dict_int, cluster_labels_text = conditional_clustering(
#         N_CLUSTERS, VARIANCE, CNN, DEVICE
#     )
#     print(cls_cluster_dict_int, cluster_labels_text)
# Example usage
if __name__ == "__main__":
    categories = CLASS_NAMES
    split_ratio = 0.4  # 2 zeros per cluster

    cluster_rotator = rotating_cluster_generator_shift([i for i in range(len(categories))], split_ratio)
    i=0
    for cluster, cat0, cat1 in cluster_rotator:
        if i<10:
            print(f"Cluster: {cluster}")
            print(f"Cat0: {cat0}")
            print(f"Cat1: {cat1}")
        else: 
            break
        i+=1
