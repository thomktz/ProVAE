import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    """
    Decoder module for the ProVAE.

    Transforms a latent vector into an image.

    Parameters
    ----------
    config : list
        A list of dictionaries containing the configuration for each resolution.
    latent_dim : int
        The dimensionality of the latent vector.
    """

    def __init__(self, config, latent_dim=200):
        super().__init__()
        self.config = config
        self.to_rgb_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # Create to_rgb layers and blocks for each resolution
        for i, cfg in enumerate(self.config):
            self.to_rgb_layers.append(
                nn.Conv2d(
                    in_channels=cfg["channels"] // 2, out_channels=3, kernel_size=1
                )
            )
            self.blocks.append(
                self._create_block(
                    cfg["channels"],
                    upscale=True,
                    last_block=(i == len(self.config) - 1),
                )
            )

        self.first_block_resolution = self.config[0]["resolution"] // 2
        self.fcl_output_dim = (
            self.first_block_resolution
            * self.first_block_resolution
            * self.config[0]["channels"]
        )
        self.fcl = nn.Linear(latent_dim, self.fcl_output_dim)

        self.alpha = 1.0
        self.level = 0

    def _create_block(
        self, channels: int, upscale: bool = True, last_block: bool = False
    ):
        """
        Create a block of layers for a given number of channels.

        Parameters
        ----------
        channels : int
            The number of channels in the input.
        upscale : bool
            Whether to add an upsampling layer at the start of the block.
        last_block : bool
            Whether this is the last block in the network.
        """
        # Create and return a block of layers for a given number of channels
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels // 2,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels // 2),
        )
        if last_block:
            block.add_module("tanh", nn.Tanh())
        else:
            block.add_module("leaky_relu", nn.LeakyReLU(0.2))
        if upscale:
            block = nn.Sequential(nn.Upsample(scale_factor=2, mode="nearest"), block)
        return block

    def forward(self, latent: torch.Tensor):
        """
        Forward pass through the decoder.

        Parameters
        ----------
        latent : torch.Tensor
            The latent vector tensor.
        """
        # Reshape the latent vector to match the input shape of the fully connected layer
        generated = self.fcl(latent).view(
            -1,
            self.config[0]["channels"],
            self.first_block_resolution,
            self.first_block_resolution,
        )

        if self.alpha == 1:
            for lvl in range(self.level + 1):
                generated = self.blocks[lvl](generated)
            generated = self.to_rgb_layers[self.level](generated)
            return generated

        for lvl in range(self.level + 1):
            generated = self.blocks[lvl](generated)
            if lvl == self.level - 1:
                prev_generated = generated
                prev_rgb = self.to_rgb_layers[lvl](prev_generated)
                upsampled_prev_rgb = F.interpolate(
                    prev_rgb, scale_factor=2, mode="nearest"
                )

        rgb = self.to_rgb_layers[self.level](generated)
        blended_rgb = self.alpha * rgb + (1 - self.alpha) * upsampled_prev_rgb

        return blended_rgb
