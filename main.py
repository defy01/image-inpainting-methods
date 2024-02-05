import torch
import sys
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import *
from utils import *

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

training_data = datasets.STL10(
    root="data", split="train", download=True, transform=transform)

test_data = datasets.STL10(root="data", split="test",
                           download=True, transform=transform)

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

# Instantiate the model and move to GPU if available
model = Autoencoder().to(device)

# Define loss function, evaluation metric and optimizer
criterion = nn.MSELoss().to(device)
dice = DiceCoef().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)

mask_size = 16
dice_weight = 0.1

# Resume training from a saved checkpoint
if opt.train:
    checkpoint = torch.load("modelhuber.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch_start = checkpoint["epoch"]
    print(f"Resuming training from epoch {epoch_start}")


elif opt.eval:
    checkpoint = torch.load("model_mse.pt")
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Model loaded for evaluation")

    model.eval()
    val_loss = 0.0
    dice_score = 0.0
    with torch.no_grad():
        for data in test_dataloader:
            inputs, _ = data
            inputs = inputs.to(device)

            # Create a center mask of size H and W of the inputs (exclude batch and channel dimensions)
            mask = create_mask(inputs.shape[2:], mask_size)
            # Expand dimensions and replicate the mask for each input in the batch
            mask = mask.unsqueeze(0).repeat(inputs.size(0), 1, 1, 1).to(device)

            # Apply mask to input images
            masked_inputs = inputs * mask

            outputs = model(masked_inputs)
            loss = criterion(outputs * (1 - mask), inputs * (1 - mask))
            val_loss += loss.item()

            dice_score += dice(outputs, inputs).item()

        avg_val_loss = val_loss / len(test_dataloader)
        avg_dice_coef = dice_score / len(test_dataloader)

        print(
            f"Validation of model_mse, Loss: {avg_val_loss:.4f}, Dice Coef: {avg_dice_coef:.4f}")

        results = model(masked_inputs)
        plt.figure(figsize=(12, 9))

        for i in range(3):
            # Plot original images
            plt.subplot(3, 3, i + 1)
            plt.imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
            plt.title("Original", fontsize=26)
            plt.axis("off")

            # Plot masked images
            plt.subplot(3, 3, i + 4)
            plt.imshow(masked_inputs[i].permute(1, 2, 0).cpu().numpy())
            plt.title("Masked", fontsize=26)
            plt.axis("off")

            # Plot reconstructed images
            inpainted_region = masked_inputs[i] + results[i] * (1 - mask)
            plt.subplot(3, 3, i + 7)
            plt.imshow(inpainted_region[0].detach().permute(
                1, 2, 0).cpu().numpy())
            plt.title("Inpainted", fontsize=26)
            plt.axis("off")

        plt.tight_layout()
        plt.savefig("evaluation/mse_loss2.svg")
        plt.show()
        plt.close()
        sys.exit()

else:
    epoch_start = 0

num_epochs = epoch_start + 50

# Training loop
for epoch in range(epoch_start, num_epochs + 1):
    model.train()
    for data in train_dataloader:
        inputs, _ = data
        inputs = inputs.to(device)

        # Create a center mask of size H and W of the inputs (exclude batch and channel dimensions)
        mask = create_mask(inputs.shape[2:], mask_size)
        # Expand dimensions and replicate the mask for each input in the batch
        mask = mask.unsqueeze(0).repeat(inputs.size(0), 1, 1, 1).to(device)

        # Apply mask to input images
        masked_inputs = inputs * mask

        optimizer.zero_grad()
        outputs = model(masked_inputs)
        # Apply mask to the loss calculation
        loss = criterion(outputs * (1 - mask), inputs * (1 - mask))

        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}")

    # Validation phase
    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        dice_score = 0.0
        for val_data in test_dataloader:
            val_inputs, _ = val_data
            val_inputs = val_inputs.to(device)

            # Create a center mask for validation set
            val_mask = create_mask(val_inputs.shape[2:], mask_size)
            val_mask = val_mask.unsqueeze(0).repeat(
                val_inputs.size(0), 1, 1, 1).to(device)

            # Apply mask to input images
            val_masked_inputs = val_inputs * val_mask

            val_outputs = model(val_masked_inputs)
            # Apply mask to the validation loss calculation
            val_loss = criterion(val_outputs * (1 - val_mask),
                                 val_inputs * (1 - val_mask))
            val_loss += val_loss.item()

            dice_score += dice(val_outputs, val_inputs).item()

        avg_val_loss = val_loss / len(test_dataloader)
        avg_dice_coef = dice_score / len(test_dataloader)

        print(
            f"Validation, Loss: {avg_val_loss:.4f}, Dice Coef: {avg_dice_coef:.4f}")

    # Plot results every X epochs
    if epoch % 10 == 0:
        results = model(masked_inputs)

        plt.figure(figsize=(15, 7))

        for i in range(5):
            # Plot original images
            plt.subplot(3, 5, i + 1)
            plt.imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
            plt.title("Original")
            plt.axis("off")

            # Plot masked images
            plt.subplot(3, 5, i + 6)
            plt.imshow(masked_inputs[i].permute(1, 2, 0).cpu().numpy())
            plt.title("Masked")
            plt.axis("off")

            # Plot reconstructed images
            inpainted_region = masked_inputs[i] + results[i] * (1 - mask)
            plt.subplot(3, 5, i + 11)
            plt.imshow(inpainted_region[0].detach().permute(
                1, 2, 0).cpu().numpy())
            plt.title("Reconstructed (Inpainted)")
            plt.axis("off")

        plt.savefig(f"training/epoch_{epoch}.png")
        plt.show()
        plt.close()

# Save the model checkpoint
torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    },
    "model20dice200.pt",
)
