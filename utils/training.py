import clip
import torch
from tqdm import tqdm

from utils.datasets import CLASS_NAMES


def test_step(model, data_loader, device="cuda"):
        samples = 0.0
        cumulative_loss = 0.0
        cumulative_accuracy = 0.0

        # Set the network to evaluation mode
        model.eval()

        # Disable gradient computation (we are only testing, we do not want our model to be modified in this step!)
        pbar = tqdm(data_loader, desc="Testing", position=0, leave=True, total=len(data_loader))
        with torch.no_grad():
            # Iterate over the test set
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                # Load data into GPU
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass
                outputs = model(inputs)

                # Loss computation
                loss = cost_function(outputs, targets)

                # Fetch prediction and loss value
                samples += inputs.shape[0]
                cumulative_loss += loss.item() # Note: the .item() is needed to extract scalars from tensors
                _, predicted = outputs.max(1)

                # Compute accuracy
                cumulative_accuracy += predicted.eq(targets).sum().item()

                pbar.set_postfix(test_acc=cumulative_accuracy / samples * 100)
                pbar.update(1)

        return cumulative_loss / samples, cumulative_accuracy / samples * 100

def training_step(net, data_loader, optimizer, cost_function, device="cuda"):
    samples = 0.0
    cumulative_loss = 0.0
    cumulative_accuracy = 0.0

    # Set the network to training mode
    net.train()

    # Iterate over the training set
    pbar = tqdm(data_loader, desc="Training", position=0, leave=True, total=len(data_loader))
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        # Load data into GPU
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = net(inputs)

        # Loss computation
        loss = cost_function(outputs, targets)

        # Backward pass
        loss.backward()

        # Parameters update
        optimizer.step()

        # Gradients reset
        optimizer.zero_grad()

        # Fetch prediction and loss value
        samples += inputs.shape[0]
        cumulative_loss += loss.item()
        _, predicted = outputs.max(dim=1) # max() returns (maximum_value, index_of_maximum_value)

        # Compute training accuracy
        cumulative_accuracy += predicted.eq(targets).sum().item()

        pbar.set_postfix(train_loss=loss.item(), train_acc=cumulative_accuracy / samples * 100)
        pbar.update(1)

    return cumulative_loss / samples, cumulative_accuracy / samples * 100


@torch.no_grad()  # we don't want gradients
def eval(model, dataset, categories, batch_size, device, label=""):
    # let's set the model in evaluation mode
    model.eval()

    # Remap labels into a contiguous set starting from zero
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}
    """
      novel_categories = [5,6,7,8,9]
    {
      5:0,
      6:1,
      7:2,
      8:3,
      9:4,
    }

    """
    # here we apply the standard CLIP template used for oxford flowers to all categories
    # and immediately tokenize each sentence (convert natural language into numbers - feel free to print the text input to inspect them)
    text_inputs = clip.tokenize(
        [f"a photo of a {CLASS_NAMES[c]}, a type of flower." for c in categories]
    ).to(device)

    # we can encode the text features once as they are shared for all images
    # therefore we do it outside the evaluation loop
    text_features = model.encode_text(text_inputs)
    # and here we normalize them (standard pratice with CLIP)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # simple dataloader creation
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # here we store the number of correct predictions we will make
    correct_predictions = 0
    for image, target in tqdm(dataloader, desc=label):
        # base categories range from 0 to 50, while novel ones from 51 to 101
        # therefore we must map categories to the [0, 50], otherwise we will have wrong predictions
        # Map targets in contiguous set starting from zero
        # Labels needs to be .long() in pytorch
        target = torch.Tensor([contig_cat2idx[t.item()] for t in target]).long()

        image = image.to(device)
        target = target.to(device)

        # forward image through CLIP image encoder
        image_features = model.encode_image(image)
        # and normalize
        image_features /= image_features.norm(dim=-1, keepdim=True)

        # here cosine similarity between image and text features and keep the argmax for every row (every image)
        predicted_class = (image_features @ text_features.T).argmax(dim=-1)
        # now we check which are correct, and sum them (False == 0, True == 1)
        correct_predictions += (predicted_class == target).sum().item()

    # and now we compute the accuracy
    accuracy = correct_predictions / len(dataset)
    return accuracy