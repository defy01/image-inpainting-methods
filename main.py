import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import *
from utils import *

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = transforms.Compose([
    transforms.ToTensor(),
])
training_data = datasets.STL10(root='data', split='train', download=True, transform=transform)
train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)

# Instantiate the model and move to GPU if available
model = Autoencoder().to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Resuming training from a saved checkpoint
if (opt.resume_training):
    checkpoint = torch.load('model.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch_start = checkpoint['epoch']
else:
    epoch_start = 0

num_epochs = epoch_start + 100
mask_size = 32
model.train()

# Training loop
for epoch in range(epoch_start, num_epochs + 1):
    for data in train_dataloader:
        inputs, _ = data
        inputs = inputs.to(device)  # Move data to GPU if available

        # Create a center mask
        mask = create_center_mask(inputs.shape[2:], mask_size)
        mask = mask.unsqueeze(0).repeat(inputs.size(0), 1, 1, 1).to(device)

        # Apply mask to input images
        masked_inputs = inputs * mask

        optimizer.zero_grad()
        outputs = model(masked_inputs)
        # Apply mask to the loss calculation
        loss = criterion(outputs * (1 - mask), inputs * (1 - mask))

        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch}/{num_epochs}], Loss: {loss.item():.4f}')

    # Plot results every X epochs
    if epoch % 10 == 0:
        sample = model(masked_inputs)

        plt.figure(figsize=(15, 7))

        for i in range(5):
            # Plot original images
            plt.subplot(3, 5, i + 1)
            plt.imshow(inputs[i].permute(1, 2, 0).cpu().numpy())
            plt.title('Original')
            plt.axis('off')

            # Plot masked images
            plt.subplot(3, 5, i + 6)
            plt.imshow(masked_inputs[i].permute(1, 2, 0).cpu().numpy())
            plt.title('Masked')
            plt.axis('off')

            # Plot reconstructed images
            inpainted_region = masked_inputs[i] + sample[i] * (1 - mask)
            plt.subplot(3, 5, i + 11)
            plt.imshow(inpainted_region[0].detach().permute(
                1, 2, 0).cpu().numpy())
            plt.title('Reconstructed (Inpainted)')
            plt.axis('off')

        plt.savefig(f'training/epoch_{epoch}.png')
        plt.show()
        plt.close()

# Save the model checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'model.pt')
