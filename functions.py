import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import skimage
import torch
import torchvision
from tqdm import tqdm



# Create a description for some images
DESCRIPTIONS = {
    "page": "a page of text about segmentation",
    "chelsea": "a facial photo of a tabby cat",
    "astronaut": "a portrait of an astronaut with the American flag",
    "rocket": "a rocket standing on a launchpad",
    "motorcycle_right": "a red motorcycle standing in a garage",
    "camera": "a person looking at a camera on a tripod",
    "horse": "a black-and-white silhouette of a horse",
    "coffee": "a cup of coffee on a saucer"
}

def get_data():
    images = []
    texts = []

    # Get all the filenames in the data directory
    data_dir = Path(skimage.data_dir)
    filenames = [
        filename for filename in data_dir.glob('*')
            if filename.suffix in {'.png', '.jpg'}
        ]

    for filename in filenames:
        # Skip images we do not care about
        name = filename.stem
        if name not in DESCRIPTIONS:
            continue

        images.append(filename)
        texts.append(DESCRIPTIONS[name])

    return images, texts

def visualise_data(images_path, texts):
    plt.figure(figsize=(16, 5))

    for i, (image_path, text) in enumerate(zip(images_path, texts)):
        # Load and visualize the image along with its text description
        image = Image.open(image_path).convert("RGB")

        plt.subplot(2, 4, i+1)
        plt.imshow(image)
        plt.title(text)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()

    plt.show()

DATASETS = {
    "mnist": torchvision.datasets.MNIST,
    "cifar10": torchvision.datasets.CIFAR10,
}

def test_step_zero_shot_clip(net, data_loader, texts_z, device='mps'):
    samples = 0.0
    cumulative_accuracy = 0.0

    # Set the network to evaluation mode
    net.eval()

    with torch.no_grad():
        # Iterate over the test set
        for batch_idx, (inputs, targets) in tqdm(enumerate(data_loader), total=len(data_loader), position=0, leave=True):
            # Load data into GPU
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            images_z = net.encode_image(inputs).float()
            images_z /= images_z.norm(dim=-1, keepdim=True)
            outputs = (100 * images_z @ texts_z.T).softmax(dim=-1)

            # Fetch prediction and loss value
            samples += inputs.shape[0]
            _, predicted = outputs.max(1)

            # Compute accuracy
            cumulative_accuracy += predicted.eq(targets).sum().item()

    return cumulative_accuracy / samples * 100

def get_optimizer(model, lr, wd, momentum):
    optimizer = torch.optim.SGD([
        {"params": model.parameters()}
    ], lr=lr, weight_decay=wd, momentum=momentum)

    return optimizer

if __name__ == "__main__":
    images_path, texts = get_data()
    print(images_path)
    visualise_data(images_path, texts)

    # Get the text descriptions and their embeddings
    classes, texts, texts_z = embed_dataset_classnames("cifar10")
    print(f"Classes: {classes}")
    print(f"Prompts (text): {texts}")
    print(f"Prompts (embedded): {texts_z.shape}")

    # Evaluate the softmax from the cosine similarities
    similarity = cosine_similarity(images_z, texts_z)
    texts_p = (100 * similarity).softmax(dim=-1)

    # Visualise the similarity
    visualise_similarity(similarity, images_path, texts)

    # Visualise the top-5 predictions
    # visualise_probabilities(images_path, texts, texts_p, k=5)
