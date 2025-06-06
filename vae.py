import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class IsingVAE(nn.Module):
    def __init__(self, input_size=32, latent_dim=2, out_channels=32, kernel_size=3, stride=1):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Compute spatial size after encoder convolutions
        self.final_spatial_size = input_size // 4  # Due to two stride=2 layers
        self.flat_size = (out_channels * 2) * self.final_spatial_size * self.final_spatial_size

        # Encoder: Input = [batch, 1, input_size, input_size]
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(self._compute_flattened_size(input_size), latent_dim)
        self.fc_logvar = nn.Linear(self._compute_flattened_size(input_size), latent_dim)

        # Decoder layers: Input = [batch, latent_dim + 1]
        self.decoder_linear = nn.Linear(latent_dim + 1, self.flat_size)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(out_channels, 1, kernel_size=kernel_size, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

    def _compute_flattened_size(self, input_size):
        with torch.no_grad():
            tmp_input = torch.randn(1, 1, input_size, input_size)
            output = self.encoder_conv(tmp_input)  # Up to flatten
            return output.numel()

    def reparameterize(self, mu, logvar):
        # Sample z = mu + sigma * epsilon, where epsilon ~ N(0, 1)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, temp):
        # x: [batch, 1, input_size, input_size], temp: [batch, 1] (normalized)
        # Encoder
        x = self.encoder_conv(x)  # [batch, flat_size]
        mu = self.fc_mu(x)  # [batch, latent_dim]
        logvar = self.fc_logvar(x)  # [batch, latent_dim]
        z = self.reparameterize(mu, logvar)  # [batch, latent_dim]

        # Decoder
        dec_input = torch.cat((z, temp), dim=1)  # [batch, latent_dim + 1]
        x = self.decoder_linear(dec_input)  # [batch, flat_size]
        x = x.view(-1, self.out_channels * 2, self.final_spatial_size, self.final_spatial_size)  # [batch, out_channels * 2, input_size // 4, input_size // 4]
        x_recon = self.decoder_conv(x)  # [batch, 1, input_size, input_size]
        return x_recon, mu, logvar, z

    def generate(self, temp, num_samples=1):
        # Generate new samples for given temperature
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(temp.device)  # [num_samples, latent_dim]
            temp = temp.expand(num_samples, 1)  # [num_samples, 1]
            dec_input = torch.cat((z, temp), dim=1)  # [num_samples, latent_dim + 1]
            x = self.decoder_linear(dec_input)  # [num_samples, flat_size]
            x = x.view(-1, self.out_channels * 2, self.final_spatial_size, self.final_spatial_size)  # [num_samples, out_channels * 2, input_size // 4, input_size // 4]
            x_gen = self.decoder_conv(x)  # [num_samples, 1, input_size, input_size]
            x_gen = (x_gen > 0.5).float() * 2 - 1  # Threshold to {-1, 1}
        return x_gen

    def loss_function(self, x_recon, x, mu, logvar, beta=1.0):
        # Reconstruction loss (BCE)
        x = (x + 1) / 2  # Map {-1, 1} to [0, 1]
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
        # KL-divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return recon_loss + beta * kld_loss, recon_loss, kld_loss