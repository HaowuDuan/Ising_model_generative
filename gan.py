import torch
import torch.nn as nn
import torch.nn.functional as F

class IsingGAN(nn.Module):
    def __init__(self, input_size=32, latent_dim=2, out_channels=32, kernel_size=3, stride=1):
        super().__init__()
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        # Compute spatial size for generator
        self.final_spatial_size = input_size // 4  # Due to two stride=2 layers
        self.flat_size = (out_channels * 2) * self.final_spatial_size * self.final_spatial_size

        # Generator: Input = [batch, latent_dim + 1] (noise + temp)
        self.generator_linear = nn.Linear(latent_dim + 1, self.flat_size)
        self.generator_conv = nn.Sequential(
            nn.ConvTranspose2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(out_channels * 2, out_channels, kernel_size=kernel_size, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(0.01),
            nn.ConvTranspose2d(out_channels, 1, kernel_size=kernel_size, stride=1, padding=1),
            nn.Sigmoid()  # Output in [0, 1]
        )

        # Discriminator: Input = [batch, 1, input_size, input_size]
        self.discriminator = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=kernel_size, stride=stride, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels, out_channels * 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=2, padding=1),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=kernel_size, stride=1, padding=1),
            nn.LeakyReLU(0.01),
            nn.Flatten(),
            nn.Linear(self._compute_flattened_size(input_size), 1),
            nn.Sigmoid()  # Probability of being real
        )

    def _compute_flattened_size(self, input_size):
        with torch.no_grad():
            tmp_input = torch.randn(1, 1, input_size, input_size)
            output = nn.Sequential(*self.discriminator[:-2])(tmp_input)  # Up to flatten
            return output.numel()

    def generate(self, temp, num_samples=1):
        # Generate samples for given temperature
        self.eval()
        with torch.no_grad():
            z = torch.randn(num_samples, self.latent_dim).to(temp.device)  # [num_samples, latent_dim]
            temp = temp.expand(num_samples, 1)  # [num_samples, 1]
            x = self.generator_forward(z, temp)  # [num_samples, 1, input_size, input_size]
            x_gen = (x > 0.5).float() * 2 - 1  # Threshold to {-1, 1}
        return x_gen

    def generator_forward(self, z, temp):
        # Forward pass for generator
        dec_input = torch.cat((z, temp), dim=1)  # [batch, latent_dim + 1]
        x = self.generator_linear(dec_input)  # [batch, flat_size]
        x = x.view(-1, self.out_channels * 2, self.final_spatial_size, self.final_spatial_size)  # [batch, out_channels * 2, input_size // 4, input_size // 4]
        x = self.generator_conv(x)  # [batch, 1, input_size, input_size]
        return x

    def discriminator_forward(self, x):
        # Forward pass for discriminator
        return self.discriminator(x)  # [batch, 1]

    def loss_function(self, x_real, x_fake, device, label_smoothing=0.1):
        # Discriminator loss
        real_labels = torch.full((x_real.size(0), 1), 1.0 - label_smoothing, device=device)
        fake_labels = torch.zeros((x_fake.size(0), 1), device=device)
        d_real = self.discriminator_forward(x_real)
        d_fake = self.discriminator_forward(x_fake.detach())
        d_loss_real = F.binary_cross_entropy(d_real, real_labels)
        d_loss_fake = F.binary_cross_entropy(d_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        # Generator loss
        g_labels = torch.ones((x_fake.size(0), 1), device=device)
        g_fake = self.discriminator_forward(x_fake)
        g_loss = F.binary_cross_entropy(g_fake, g_labels)

        return d_loss, g_loss