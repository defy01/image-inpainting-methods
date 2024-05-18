"""
Author: Tomáš Rajsigl
Email: xrajsi01@stud.fit.vutbr.cz
Filename: main.py

This script defines a Trainer class for training an image inpainting model using PyTorch. The Trainer
class includes methods for starting training, saving checkpoints during training, and resuming training
from a checkpoint. It also defines an Evaluator class for evaluating the trained model using standard metrics. 
"""

import torch
import sys
import argparse
import wandb
import torch.nn as nn
import torch.optim as optim

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from model import *
from aotgan import *
from utils import *


class Trainer:
    """
    Trainer class for training an image inpainting model.

    Attributes:
        model (torch.nn.Module): The PyTorch model.
        criterion (torch.nn.Module): The criterion for Reconstruction loss.
        feature_criterion (torch.nn.Module): The feature extraction module.
        optimizer (torch.optim.Optimizer): The optimizer.
        scheduler (torch.optim.lr_scheduler._LRScheduler): The generator learning rate scheduler.
        train_dataloader (torch.utils.data.DataLoader): The training data loader.
        test_dataloader (torch.utils.data.DataLoader): The test data loader.
        mask_dataloader (torch.utils.data.DataLoader): The mask data loader.
        device (str): The device for training (e.g., 'cpu', 'cuda').
    """

    def __init__(
        self,
        model,
        criterion,
        feature_criterion,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
        mask_dataloader,
        device,
    ):
        self.model = model
        self.discriminator = Discriminator().to(device)
        self.criterion = criterion
        self.adv_loss = nn.MSELoss().to(device) # Discriminator optimizes MSE instead of BCE (LSGAN)
        self.feature_criterion = feature_criterion
        self.optimizer_G = optimizer
        self.optimizer_D = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.scheduler_G = scheduler
        self.scheduler_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_D, factor=0.5, patience=6, min_lr=1e-6, verbose=True
        )
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.mask_dataloader = mask_dataloader
        self.device = device
        self.visualizer = Visualizer(plot_metrics=False)

    def save(self, epoch, loss, checkpoint_path):
        """
        Saves a checkpoint of the model and optimizer state during training.

        Args:
            epoch (int): The current epoch number.
            loss (float): The value of the loss function at the checkpoint.
            checkpoint_path (str): The file path where the checkpoint will be saved.
        """
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer_G.state_dict(),
                "loss": loss,
            },
            checkpoint_path,
        )
        print(f"Model checkpoint saved to {checkpoint_path}.")

    def start(self, num_epochs, path, resume=False):
        """
        Starts the training process of the inpainting model.

        Args:
            num_epochs (int): The total number of epochs to training for.
            path (str): The file path to save the model or to resume training from if 'resume' is True.
            resume (bool, optional): Whether to resume training from a checkpoint (default is False).
        """
        if resume:
            self.early_stopper = EarlyStopping(patience=20, verbose=True, path="model.pt")
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer_G.load_state_dict(checkpoint["optimizer_state_dict"])
            epoch_start = checkpoint["epoch"]
            print(f"Resuming training, starting from epoch {epoch_start}")
        else:
            self.early_stopper = EarlyStopping(patience=20, verbose=True, path=path)
            epoch_start = 1

        # start a new wandb run to track training
        wandb.init(
            project="image-inpainting",
            config={
                "learning_rate": "0.0002",
                "architecture": "ResUNet",
                "vgg_architecture": "VGG19",
                "loss": "L1 + Perceptual + Style + Adversarial",
                "lambda_reconstruction": 1,
                "lambda_perceptual": 0.1,
                "lambda_style": 120,
                "lambda_adversarial": 0.01,
                "dataset": "Places365",
                "mask dataset": "NVIDIA Irregular Mask Dataset",
                "epochs": 300,
            },
        )

        # Training phase
        try:
            for epoch in range(epoch_start, num_epochs + 1):
                self.model.train()
                self.discriminator.train()
                train_loss = torch.tensor(0.0, device=self.device)
                for data, mask_data in zip(self.train_dataloader, self.mask_dataloader):
                    inputs, _ = data
                    masks = mask_data.to(self.device)
                    inputs = inputs.to(self.device)

                    masked_inputs = (inputs * (1 - masks)) + masks
                    outputs = self.model(masked_inputs, masks)
                    inpainted_image = outputs * masks + inputs * (1 - masks)

                    ## Discriminator
                    real_labels = torch.ones(inputs.size(0), 1, 30, 30, device=self.device)
                    fake_labels = torch.zeros(inputs.size(0), 1, 30, 30, device=self.device)

                    # Discriminator loss for real samples
                    D_real = self.discriminator(inputs)
                    D_real_loss = self.adv_loss(D_real, real_labels)

                    # Discriminator loss for fake samples
                    D_fake = self.discriminator(inpainted_image.detach())
                    D_fake_loss = self.adv_loss(D_fake, fake_labels)

                    D_loss = (D_real_loss + D_fake_loss) / 2

                    self.optimizer_D.zero_grad()
                    D_loss.backward()
                    self.optimizer_D.step()

                    ## Generator
                    gen_output = self.discriminator(inpainted_image)
                    adv_loss = self.adv_loss(gen_output, real_labels)
                    rec_loss = self.criterion(outputs * masks, inputs * masks)
                    perc_loss, style_loss = self.feature_criterion(inputs, inpainted_image)
                    loss = rec_loss + 0.1 * perc_loss + 120 * style_loss + 0.01 * adv_loss
                    train_loss += loss

                    self.optimizer_G.zero_grad()
                    loss.backward()
                    self.optimizer_G.step()

                avg_gen_loss = train_loss / len(self.mask_dataloader)
                avg_disc_loss = D_loss / len(self.mask_dataloader)
                wandb.log({"Generator loss": avg_gen_loss, "Discriminator loss": avg_disc_loss}, commit=False)
                print(f"Epoch [{epoch}/{num_epochs}], L1 + Perceptual + Style + Adv Loss: {avg_gen_loss:.4f}")

                # Validation phase
                self.model.eval()
                val_loss = torch.tensor(0.0, device=self.device)
                with torch.no_grad():
                    for data, mask_data in zip(self.test_dataloader, self.mask_dataloader):
                        inputs, _ = data
                        masks = mask_data.to(self.device)
                        inputs = inputs.to(self.device)

                        masked_inputs = (inputs * (1 - masks)) + masks
                        outputs = self.model(masked_inputs, masks)
                        inpainted_image = outputs * masks + inputs * (1 - masks)

                        loss = self.criterion(outputs * masks, inputs * masks)
                        perc_loss, style_loss = self.feature_criterion(inputs, inpainted_image)
                        loss = loss + 0.1 * perc_loss + 120 * style_loss
                        val_loss += loss

                    avg_val_loss = val_loss / len(self.mask_dataloader)
                    wandb.log({"Validation loss": avg_val_loss})
                    print(f"Validation, Loss: {avg_val_loss:.4f}")

                self.scheduler_G.step(avg_val_loss)
                self.scheduler_D.step(avg_val_loss)
                self.early_stopper(avg_val_loss, self.model)

                if self.early_stopper.early_stop:
                    print("Early stopping")
                    break

                # Plot results every X epochs
                if epoch % 20 or epoch == 1:
                    self.visualizer.plot(inputs, masked_inputs, outputs, masks, f"training/epoch_{epoch}.pdf")
        
        except KeyboardInterrupt:
            print("Training interrupted. Saving model...")
            self.save(epoch, loss, "interrupted_model.pt")


class Evaluator:
    """
    Evaluator class for evaluating the trained image inpainting model.

    Attributes:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        criterion (torch.nn.Module): The loss criterion.
        test_dataloader (torch.utils.data.DataLoader): The test data loader.
        mask_dataloader (torch.utils.data.DataLoader): The mask data loader.
        device (str): The device for evaluation (e.g., 'cpu', 'cuda').
    """

    def __init__(self, model, criterion, test_dataloader, mask_dataloader, device):
        self.model = model
        self.criterion = criterion
        self.test_dataloader = test_dataloader
        self.mask_dataloader = mask_dataloader
        self.device = device
        self.visualizer = Visualizer(plot_metrics=True)

    def evaluate(self, checkpoint_path):
        """
        Loads the model from the provided 'checkpoint_path' for evaluation.
        Computes evaluation metrics such as L1 loss, SSIM, PSNR, and LPIPS score on the test dataset.

        Args:
            checkpoint_path (str): The file path to the model to be evaluated.
        """
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        print(f"Model {checkpoint_path} loaded for evaluation")

        self.model.eval()
        val_loss = torch.tensor(0.0, device=self.device)
        val_lpips = torch.tensor(0.0, device=self.device)
        batch_ssim = torch.tensor(0.0, device=self.device)
        batch_psnr = torch.tensor(0.0, device=self.device)
        lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="sum", normalize=True)
        lpips = lpips.to(self.device)

        with torch.no_grad():
            for data, mask_data in zip(self.test_dataloader, self.mask_dataloader):
                inputs, _ = data
                masks = mask_data.to(self.device)
                inputs = inputs.to(self.device)

                masked_inputs = (inputs * (1 - masks)) + masks
                outputs = self.model(masked_inputs, masks)
                inpainted_image = outputs * masks + inputs * (1 - masks)

                loss = self.criterion(outputs * masks, inputs * masks)
                val_loss += loss
                lpips_score = lpips(inputs, inpainted_image)
                val_lpips += lpips_score

                # Calculate SSIM and PSNR for each image in the batch
                for i in range(inputs.size(0)):
                    ssim_score = ssim(
                        inputs[i].permute(1, 2, 0).cpu().numpy(),
                        inpainted_image[i].permute(1, 2, 0).detach().cpu().numpy(),
                        win_size=1,
                        channel_axis=2,
                        data_range=1.0,
                        gaussian_weights=True,
                        sigma=1.5,
                        use_sample_covariance=False,
                    )
                    psnr_score = psnr(
                        inputs[i].permute(1, 2, 0).cpu().numpy(),
                        inpainted_image[i].permute(1, 2, 0).detach().cpu().numpy(),
                        data_range=1,
                    )

                    batch_ssim += ssim_score
                    batch_psnr += psnr_score

            avg_val_loss = val_loss / len(self.mask_dataloader)
            avg_lpips = val_lpips / len(self.mask_dataloader.sampler)
            avg_ssim = batch_ssim / len(self.mask_dataloader.sampler)
            avg_psnr = batch_psnr / len(self.mask_dataloader.sampler)

            print(f"L1 Loss: {avg_val_loss:.4f}")
            print(f"Average SSIM: {avg_ssim:.4f}")
            print(f"Average PSNR: {avg_psnr:.4f}")
            print(f"Average LPIPS: {avg_lpips:.4f}")

            self.visualizer.plot(inputs, masked_inputs, outputs, masks, f"evaluation/{checkpoint_path}.pdf")
            sys.exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-t", "--train", type=str, help="start training, specify the path of the PyTorch model to be saved")
    group.add_argument("-r", "--resume", type=str, help="resume training from a saved checkpoint")
    group.add_argument("-e", "--eval", type=str, help="run evaluation of the model from a saved checkpoint")
    parser.add_argument("-b", "--batch_size", type=int, default=40, help="batch size for training (default: 40)")
    opt = parser.parse_args()
    print(vars(opt))

    transform, mask_transform, shuffle_flag = prepare_transforms(opt)
    train_dataloader, test_dataloader, mask_dataloader = prepare_data(transform, mask_transform, shuffle_flag, opt.batch_size)

    # Instantiate the model and move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ResidualUNet().to(device)

    # Define loss functions and optimizer
    criterion = nn.L1Loss().to(device)
    feature_criterion = FeatureExtractor().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=6, min_lr=1e-6, verbose=True)

    if opt.train or opt.resume:
        trainer_args = {
            "model": model,
            "criterion": criterion,
            "feature_criterion": feature_criterion,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "train_dataloader": train_dataloader,
            "test_dataloader": test_dataloader,
            "mask_dataloader": mask_dataloader,
            "device": device,
        }
        trainer = Trainer(**trainer_args)
        resume = True if opt.resume else False
        trainer.start(num_epochs=300, path=opt.train if opt.train else opt.resume, resume=resume)

    elif opt.eval:
        evaluator_args = {
            "model": model,
            "criterion": criterion,
            "test_dataloader": test_dataloader,
            "mask_dataloader": mask_dataloader,
            "device": device,
        }
        evaluator = Evaluator(**evaluator_args)
        evaluator.evaluate(opt.eval)
