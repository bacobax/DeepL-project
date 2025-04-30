import clip
import torch
from tqdm import tqdm

from utils.datasets import CLASS_NAMES



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


@torch.no_grad()
def test_step(model, dataset, categories, batch_size, device, label=""):
    model.eval()

    # Map categories to contiguous indices
    contig_cat2idx = {cat: idx for idx, cat in enumerate(categories)}

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    correct_predictions = 0
    total = 0

    for images, targets in tqdm(dataloader, desc=label):
        images = images.to(device)
        targets = torch.tensor([contig_cat2idx[t.item()] for t in targets]).long().to(device)

        # Directly call model.forward()
        logits = model(images)  # this returns the cosine similarity scores

        predictions = logits.argmax(dim=-1)
        correct_predictions += (predictions == targets).sum().item()
        total += targets.size(0)

    accuracy = correct_predictions / total
    return accuracy