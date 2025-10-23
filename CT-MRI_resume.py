import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
import itertools
import gc
import sys
import traceback
from datetime import datetime
import random

# Logging and Utility Functions

def log_print(message, log_file='/output/training_log.txt'):
    #Log to both console and file
    print(message)
    sys.stdout.flush()
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")
        f.flush()

def psnr_loss(img1, img2, max_val=1.0):
    img1_norm = (img1 + 1.0) / 2.0
    img2_norm = (img2 + 1.0) / 2.0
    mse = torch.mean((img1_norm - img2_norm) ** 2)
    if mse == 0:
        return torch.tensor(100.0)
    return 20 * torch.log10(max_val / torch.sqrt(mse))

def psnr_loss_as_loss(img1, img2, max_val=1.0):
    return -psnr_loss(img1, img2, max_val)

def ssim_loss(img1, img2, metric):
    
    img1_norm = (img1 + 1.0) / 2.0  
    img2_norm = (img2 + 1.0) / 2.0
    ssim_val = metric(img1_norm, img2_norm)
    return 1 - ssim_val

# Image History Buffer

class ImageBuffer:
    def __init__(self, buffer_size=50, device='cpu'):
        self.buffer_size = buffer_size
        self.image_buffer = []
        self.device = device

    def push_and_pop(self, images):
        if self.buffer_size == 0:
            return images
        
        to_return = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if len(self.image_buffer) < self.buffer_size:
                self.image_buffer.append(image)
                to_return.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_index = random.randint(0, self.buffer_size - 1)
                    old_image = self.image_buffer[random_index].clone()
                    self.image_buffer[random_index] = image
                    to_return.append(old_image)
                else:
                    to_return.append(image)
        return torch.cat(to_return, 0).to(self.device)

# Dataset

class cycleganDataset(Dataset):
    def __init__(self, trainA_path, trainB_path, transform=None):
        self.trainA_images = sorted([os.path.join(trainA_path, img) for img in os.listdir(trainA_path)])
        self.trainB_images = sorted([os.path.join(trainB_path, img) for img in os.listdir(trainB_path)])
        self.transform = transform

    def __len__(self):
        return min(len(self.trainA_images), len(self.trainB_images))

    def __getitem__(self, idx):
        img_A = Image.open(self.trainA_images[idx]).convert('RGB')
        img_B = Image.open(self.trainB_images[idx]).convert('RGB')
        if self.transform:
            img_A = self.transform(img_A)
            img_B = self.transform(img_B)
        return img_A, img_B

# Network Architectures

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.block(x) + x)

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=6):
        super(Generator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        self.residual_blocks = nn.Sequential(*[ResidualBlock(512) for _ in range(num_residual_blocks)])
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        residual_out = self.residual_blocks(encoded)
        return self.decoder(residual_out)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

# Helper Functions

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_metrics_to_csv(epoch, d_loss, g_loss, cycle_loss, id_loss, psnr_loss, psnr_val, ssim_loss,
                        csv_file='/output/training_metrics.csv'):
    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    file_exists = os.path.isfile(csv_file)
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Epoch', 'D_Loss', 'G_Loss', 'Cycle_Loss', 'ID_Loss', 'PSNR_Loss', 'PSNR_dB', 'SSIM_Loss'])
        writer.writerow([epoch, d_loss, g_loss, cycle_loss, id_loss, psnr_loss, psnr_val, ssim_loss])

def find_latest_checkpoint(save_dir):
    if not os.path.exists(save_dir):
        return None, 0
    
    checkpoint_files = [f for f in os.listdir(save_dir) if f.endswith('.pth')]
    if not checkpoint_files:
        return None, 0
   
    epoch_numbers = []
    for f in checkpoint_files:
        if 'G_AB_epoch_' in f:
            try:
                epoch_num = int(f.split('_')[-1].split('.')[0])
                epoch_numbers.append(epoch_num)
            except:
                continue
    
    if not epoch_numbers:
        return None, 0
    
    latest_epoch = max(epoch_numbers)
    return latest_epoch, latest_epoch

def load_checkpoint(model, optimizer, save_dir, model_name, epoch):
    checkpoint_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch:04d}.pth')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if model:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        log_print(f"Loaded checkpoint for {model_name} from epoch {epoch}")
        return True
    return False

def save_checkpoint(model, optimizer, save_dir, model_name, epoch):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }
    checkpoint_path = os.path.join(save_dir, f'{model_name}_epoch_{epoch:04d}.pth')
    torch.save(checkpoint, checkpoint_path)

# Training Function (Modified)

def main():
    try:
        # Configuration Dictionary
        config = {
            'learning_rate': 0.0002,
            'betas': (0.5, 0.999),
            'num_epochs': 500,
            'decay_start_epoch': 100,
            'lambda_cycle': 10.0,
            'lambda_id': 0.1,
            'lambda_psnr': 2.0,
            'lambda_ssim': 10.0,
            'batch_size': 4,
            'buffer_size': 50,
            'image_size': 256,
            'trainA_path': 'DATASET/images/trainA', 
            'trainB_path': 'DATASET/images/trainB', 
            'save_dir': 'output/checkpoints',
            'log_file': 'output/training_log.txt',
            'metrics_file': 'output/training_metrics.csv'
        }

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        log_print(f"Using device: {device}", log_file=config['log_file'])

        if torch.cuda.is_available():
            log_print(f"GPU: {torch.cuda.get_device_name(0)}", log_file=config['log_file'])
            log_print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", log_file=config['log_file'])

        os.makedirs(config['save_dir'], exist_ok=True)
        os.makedirs('output/saved_images', exist_ok=True)

        transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.RandomHorizontalFlip(p=0.5),  # Add data augmentation
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset = cycleganDataset(config['trainA_path'], config['trainB_path'], transform)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

        G_AB, G_BA = Generator().to(device), Generator().to(device)
        D_A, D_B = Discriminator().to(device), Discriminator().to(device)

        # Find latest checkpoint and load if exists
        latest_epoch, epoch_offset = find_latest_checkpoint(config['save_dir'])
        start_epoch = 1
        
        if latest_epoch > 0:
            log_print(f"Found checkpoint from epoch {latest_epoch}. Resuming training...", log_file=config['log_file'])
            start_epoch = latest_epoch + 1
            
            # Load generator checkpoints
            load_checkpoint(G_AB, None, config['save_dir'], 'G_AB', latest_epoch)
            load_checkpoint(G_BA, None, config['save_dir'], 'G_BA', latest_epoch)
            
            # Load discriminator checkpoints  
            load_checkpoint(D_A, None, config['save_dir'], 'D_A', latest_epoch)
            load_checkpoint(D_B, None, config['save_dir'], 'D_B', latest_epoch)
            
        else:
            log_print("No checkpoint found. Initializing weights from scratch.", log_file=config['log_file'])
            G_AB.apply(weights_init)
            G_BA.apply(weights_init)
            D_A.apply(weights_init)
            D_B.apply(weights_init)

        criterion_gan = nn.BCEWithLogitsLoss()
        criterion_l1 = nn.L1Loss()
        
        # Initialize image buffers for discriminators
        fake_A_buffer = ImageBuffer(config['buffer_size'], device=device)
        fake_B_buffer = ImageBuffer(config['buffer_size'], device=device)

        opt_G = optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()), 
                           lr=config['learning_rate'], betas=config['betas'])
        opt_D_A = optim.Adam(D_A.parameters(), lr=config['learning_rate'], betas=config['betas'])
        opt_D_B = optim.Adam(D_B.parameters(), lr=config['learning_rate'], betas=config['betas'])

        # Load optimizer states if resuming
        if latest_epoch > 0:
            load_checkpoint(None, opt_G, config['save_dir'], 'G_AB', latest_epoch)
            load_checkpoint(None, opt_D_A, config['save_dir'], 'D_A', latest_epoch) 
            load_checkpoint(None, opt_D_B, config['save_dir'], 'D_B', latest_epoch)
        
        # Learning rate schedulers
        lr_scheduler_G = optim.lr_scheduler.LambdaLR(opt_G, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config['decay_start_epoch']) / (config['num_epochs'] - config['decay_start_epoch']))
        lr_scheduler_D_A = optim.lr_scheduler.LambdaLR(opt_D_A, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config['decay_start_epoch']) / (config['num_epochs'] - config['decay_start_epoch']))
        lr_scheduler_D_B = optim.lr_scheduler.LambdaLR(opt_D_B, 
            lr_lambda=lambda epoch: 1.0 - max(0, epoch - config['decay_start_epoch']) / (config['num_epochs'] - config['decay_start_epoch']))
        
        ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

        log_print("Starting training.", log_file=config['log_file'])

        for epoch in range(start_epoch, config['num_epochs'] + 1):
            for step, (real_A, real_B) in enumerate(dataloader):
                real_A, real_B = real_A.to(device), real_B.to(device)

                fake_A = G_BA(real_B)
                fake_B = G_AB(real_A)

                # Train Discriminators
                opt_D_A.zero_grad()
                fake_A_from_buffer = fake_A_buffer.push_and_pop(fake_A)
                loss_D_A_real = criterion_gan(D_A(real_A), torch.ones_like(D_A(real_A)))
                loss_D_A_fake = criterion_gan(D_A(fake_A_from_buffer.detach()), torch.zeros_like(D_A(fake_A_from_buffer)))
                loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)
                loss_D_A.backward()
                opt_D_A.step()

                opt_D_B.zero_grad()
                fake_B_from_buffer = fake_B_buffer.push_and_pop(fake_B)
                loss_D_B_real = criterion_gan(D_B(real_B), torch.ones_like(D_B(real_B)))
                loss_D_B_fake = criterion_gan(D_B(fake_B_from_buffer.detach()), torch.zeros_like(D_B(fake_B_from_buffer)))
                loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)
                loss_D_B.backward()
                opt_D_B.step()
                
                loss_D_total = loss_D_A + loss_D_B

                # Train Generators
                opt_G.zero_grad()
                
                # GAN Loss
                loss_G_adv = criterion_gan(D_B(fake_B), torch.ones_like(D_B(fake_B))) + \
                             criterion_gan(D_A(fake_A), torch.ones_like(D_A(fake_A)))

                # Cycle Consistency Loss
                rec_A = G_BA(fake_B)
                rec_B = G_AB(fake_A)
                loss_cycle = criterion_l1(rec_A, real_A) + criterion_l1(rec_B, real_B)
                
                # Identity Loss
                loss_id = criterion_l1(G_BA(real_A), real_A) + criterion_l1(G_AB(real_B), real_B)
                
                # PSNR Loss
                loss_psnr = psnr_loss_as_loss(rec_A, real_A) + psnr_loss_as_loss(rec_B, real_B)
                
                # SSIM Loss
                loss_ssim = ssim_loss(rec_A, real_A, ssim_metric) + ssim_loss(rec_B, real_B, ssim_metric)

                # Total Generator Loss
                loss_G_total = (
                    loss_G_adv
                    + config['lambda_cycle'] * loss_cycle
                    + config['lambda_id'] * loss_id
                    + config['lambda_psnr'] * loss_psnr
                    + config['lambda_ssim'] * loss_ssim
                )

                # Backpropagation
                loss_G_total.backward()
                torch.nn.utils.clip_grad_norm_(itertools.chain(G_AB.parameters(), G_BA.parameters()), max_norm=1.0)
                opt_G.step()
            
            if epoch % 50 == 0:
                with torch.no_grad():
                    fake_B_sample = fake_B.detach()
                    fake_A_sample = fake_A.detach()
                    rec_A_sample = rec_A.detach()
                    rec_B_sample = rec_B.detach()

                    # Rescale from [-1,1] to [0,1]
                    def denorm(x):
                        return torch.clamp((x * 0.5) + 0.5, 0, 1)

                    # Combine images horizontally: real_CT | fake_MRI | rec_CT | real_MRI | fake_CT | rec_MRI
                    grid = torch.cat([
                        denorm(real_A),
                        denorm(fake_B_sample),
                        denorm(rec_A_sample),
                        denorm(real_B),
                        denorm(fake_A_sample),
                        denorm(rec_B_sample)
                    ], dim=3)  

                    save_path = f"output/saved_images/composite_epoch_{epoch}.png"
                    save_image(grid.cpu(), save_path)
                    log_print(f"Saved composite image grid at {save_path}", log_file=config['log_file'])


            # Update learning rates
            lr_scheduler_G.step()
            lr_scheduler_D_A.step()
            lr_scheduler_D_B.step()
            
            # Calculate PSNR for logging
            psnr_A = psnr_loss(rec_A, real_A).item()
            psnr_B = psnr_loss(rec_B, real_B).item()
            psnr_val = (psnr_A + psnr_B) / 2

            log_print(
                f"Epoch [{epoch}/{config['num_epochs']}] - G_loss: {loss_G_total.item():.4f}, D_loss: {loss_D_total.item():.4f}", 
                log_file=config['log_file']
            )
            log_print(
                f"  Cycle Loss: {loss_cycle.item():.4f}, ID Loss: {loss_id.item():.4f}, PSNR Loss (as loss): {loss_psnr.item():.4f}, PSNR (dB): {psnr_val:.2f}, SSIM: {loss_ssim.item():.4f}",
                log_file=config['log_file']
            )
            save_metrics_to_csv(
                epoch, loss_D_total.item(), loss_G_total.item(), loss_cycle.item(),
                loss_id.item(), loss_psnr.item(), psnr_val, loss_ssim.item(), 
                csv_file=config['metrics_file']
            )

            if epoch % 50 == 0:
                save_checkpoint(G_AB, opt_G, config['save_dir'], 'G_AB', epoch)
                save_checkpoint(G_BA, opt_G, config['save_dir'], 'G_BA', epoch)
                save_checkpoint(D_A, opt_D_A, config['save_dir'], 'D_A', epoch)
                save_checkpoint(D_B, opt_D_B, config['save_dir'], 'D_B', epoch)
                log_print(f"Checkpoint saved at epoch {epoch}", log_file=config['log_file'])
        
            torch.cuda.empty_cache()
            gc.collect()    
        log_print("Training completed successfully!", log_file=config['log_file'])

    except Exception as e:
        log_print(f"ERROR: {e}")
        traceback.print_exc()

if __name__ == "__main__":

    main()
