import clip
import torch
import torchvision
from torch.utils.data import Subset

CLASS_NAMES = ["pink primrose", "hard-leaved pocket orchid", "canterbury bells", "sweet pea", "english marigold", "tiger lily", "moon orchid", "bird of paradise", "monkshood", "globe thistle", "snapdragon", "colt's foot", "king protea", "spear thistle", "yellow iris", "globe-flower", "purple coneflower", "peruvian lily", "balloon flower", "giant white arum lily", "fire lily", "pincushion flower", "fritillary", "red ginger", "grape hyacinth", "corn poppy", "prince of wales feathers", "stemless gentian", "artichoke", "sweet william", "carnation", "garden phlox", "love in the mist", "mexican aster", "alpine sea holly", "ruby-lipped cattleya", "cape flower", "great masterwort", "siam tulip", "lenten rose", "barbeton daisy", "daffodil", "sword lily", "poinsettia", "bolero deep blue", "wallflower", "marigold", "buttercup", "oxeye daisy", "common dandelion", "petunia", "wild pansy", "primula", "sunflower", "pelargonium", "bishop of llandaff", "gaura", "geranium", "orange dahlia", "pink-yellow dahlia?", "cautleya spicata", "japanese anemone", "black-eyed susan", "silverbush", "californian poppy", "osteospermum", "spring crocus", "bearded iris", "windflower", "tree poppy", "gazania", "azalea", "water lily", "rose", "thorn apple", "morning glory", "passion flower", "lotus", "toad lily", "anthurium", "frangipani", "clematis", "hibiscus", "columbine", "desert-rose", "tree mallow", "magnolia", "cyclamen", "watercress", "canna lily", "hippeastrum", "bee balm", "ball moss", "foxglove", "bougainvillea", "camellia", "mallow", "mexican petunia", "bromelia", "blanket flower", "trumpet creeper", "blackberry lily"]


def get_data(data_dir="./data", transform=None):
    """
    Loads the Flowers102 dataset from torchvision, returning separate splits for training, validation, and testing.

    Args:
        data_dir (str): Directory where the dataset will be downloaded/stored. Defaults to "./data".
        transform (torchvision.transforms.Compose or None): Transformations to apply to each image.

    Returns:
        tuple: A tuple (train, val, test) of Flowers102 dataset splits.
    """
    train = torchvision.datasets.Flowers102(root=data_dir, split="train", download=True, transform=transform)
    val = torchvision.datasets.Flowers102(root=data_dir, split="val", download=True, transform=transform)
    test = torchvision.datasets.Flowers102(root=data_dir, split="test", download=True, transform=transform)
    return train, val, test


def base_novel_categories(dataset):
    # set returns the unique set of all dataset classes
    all_classes = set(dataset._labels)
    # and let's count them
    num_classes = len(all_classes)

    # here list(range(num_classes)) returns a list from 0 to num_classes - 1
    # then we slice the list in half and generate base and novel category lists
    base_classes = list(range(num_classes))[:num_classes//2]
    novel_classes = list(range(num_classes))[num_classes//2:]
    return base_classes, novel_classes


def get_labels(dataset):
    """
    Recursively retrieve labels from dataset or nested Subset.
    Assumes the base dataset has a `_labels` attribute.
    """
    if hasattr(dataset, '_labels'):
        return dataset._labels
    elif isinstance(dataset, Subset):
        parent_labels = get_labels(dataset.dataset)
        return [parent_labels[i] for i in dataset.indices]
    else:
        raise AttributeError("Dataset does not have _labels or is not a Subset of a dataset with _labels.")

def split_data(dataset, base_classes):
    """
    Splits the dataset into base and novel subsets based on base_classes.
    Works even if the input dataset is already a Subset.

    Args:
        dataset: PyTorch Dataset or Subset
        base_classes (List[int]): List of class indices considered as base.

    Returns:
        base_dataset (Subset): Subset containing samples from base classes.
        novel_dataset (Subset): Subset containing samples from novel classes.
    """
    base_categories_samples = []
    novel_categories_samples = []

    labels = get_labels(dataset)
    base_set = set(base_classes)

    for sample_id, label in enumerate(labels):
        if label in base_set:
            base_categories_samples.append(sample_id)
        else:
            novel_categories_samples.append(sample_id)

    base_dataset = Subset(dataset, base_categories_samples)
    novel_dataset = Subset(dataset, novel_categories_samples)

    return base_dataset, novel_dataset


class ContiguousLabelDataset(torch.utils.data.Dataset):
    """
    A dataset wrapper that remaps arbitrary class labels to contiguous integers starting from 0.

    This is useful for classification tasks where models expect class indices to be in a 0-based contiguous range.

    Attributes:
        dataset (Dataset): The original dataset to wrap.
        cat2idx (Dict[Any, int]): Mapping from original class labels to contiguous integer indices.
        idx2cat (Dict[int, Any]): Reverse mapping from indices back to original class labels.
    """
    def __init__(self, dataset):
        self.dataset = dataset
        # Extract all labels from the dataset
        labels = [label for _, label in dataset]
        unique_labels = sorted(set(labels))
        self.cat2idx = {cat: idx for idx, cat in enumerate(unique_labels)}
        self.idx2cat = {idx: cat for cat, idx in self.cat2idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        mapped_label = self.cat2idx[label]
        return image, mapped_label

if __name__ == "__main__":
    # Load model
    clip_model, preprocess = clip.load("ViT-B/32", device="mps")
    clip_model = clip_model.to("mps")

    train_set, val_set, test_set = get_data(transform=preprocess, data_dir="../data")
    base_classes, novel_classes = base_novel_categories(train_set)
    print(len(base_classes))