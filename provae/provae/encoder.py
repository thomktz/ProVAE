import torch.nn.functional as F
from torch import nn


class Encoder(nn.Module):
    """
    Encoder network for ProVAE.

    Implemets progressive growing by blending the outputs of the blocks at different resolutions.

    Parameters
    ----------
    config : list of dicts
        The configuration for the progressive growing.
    """

    def __init__(self, config, latent_dim=200):
        super().__init__()
        self.config = config
        self.from_rgb_layers = nn.ModuleList()
        self.blocks = nn.ModuleList()

        # Create from_rgb layers and blocks for each resolution
        for cfg in self.config:
            self.from_rgb_layers.append(
                nn.Conv2d(
                    in_channels=3, out_channels=cfg["channels"] // 2, kernel_size=1
                )
            )
            self.blocks.append(self._create_block(cfg["channels"], downscale=True))

        # Fully connected layer at the end of the last block - sizes
        self.last_block_resolution = self.config[0]["resolution"]
        self.fcl_input_dim = (
            self.last_block_resolution
            * self.last_block_resolution
            * self.config[0]["channels"]
            // 4
        )

        # Two separate FCLs for mu and logvar
        self.fc_mu = nn.Linear(self.fcl_input_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.fcl_input_dim, latent_dim)

        self.alpha = 1.0
        self.level = 0

    def _create_block(self, channels, downscale=True):
        """
        Creates a block of layers for a given resolution.

        Parameters
        ----------
        channels : int
            The number of channels in the input and output tensors.
        downscale : bool (default: True)
            Whether to downscale the input tensor at the end of the block.
        """
        block = nn.Sequential(
            nn.Conv2d(
                in_channels=channels // 2,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(channels),
            nn.LeakyReLU(0.2),
        )
        if downscale:
            # Add downsampling at the end of the block
            block.add_module("downscale", nn.AvgPool2d(kernel_size=2, stride=2))
        else:
            block.add_module("downscale", nn.Identity())

        return block

    def forward(self, img):
        """
        Forward pass through the encoder.

        Parameters
        ----------
        img : torch.Tensor
            The input image tensor.
        """
        if self.alpha == 1:
            latent = self.from_rgb_layers[self.level](img)
            for lvl in range(self.level, -1, -1):
                latent = self.blocks[lvl](latent)

        else:
            latent = self.from_rgb_layers[self.level](img)
            for lvl in range(self.level, -1, -1):
                latent = self.blocks[lvl](latent)

            if self.level > 0:
                downsampled_img = F.avg_pool2d(img, kernel_size=2, stride=2)
                downsampled_latent = self.from_rgb_layers[self.level - 1](
                    downsampled_img
                )
                for lvl in range(self.level - 1, -1, -1):
                    downsampled_latent = self.blocks[lvl](downsampled_latent)
                latent = (1 - self.alpha) * downsampled_latent + self.alpha * latent

        # Reshape and pass through two separate linear layers for mu and logvar
        latent = latent.view(-1, self.fcl_input_dim)
        mu = self.fc_mu(latent)
        logvar = self.fc_logvar(latent)

        return mu, logvar
