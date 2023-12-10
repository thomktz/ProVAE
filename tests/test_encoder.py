import pytest
import torch
from provae.provae.encoder import Encoder
from provae.provae.config import config


@pytest.mark.parametrize(
    "level, alpha, input_shape, expected_shape",
    [
        (0, 1, (1, 3, 4, 4), (1, 200)),  # Level 0, no blending
        (1, 0.5, (1, 3, 8, 8), (1, 200)),  # Level 1, blending
        (1, 1, (1, 3, 8, 8), (1, 200)),  # Level 1, no blending
        (2, 0.5, (1, 3, 16, 16), (1, 200)),  # Level 2, blending
        (2, 1.0, (1, 3, 16, 16), (1, 200)),  # Level 2, no blending
    ],
)
def test_encoder_output_shape(level, alpha, input_shape, expected_shape):
    encoder = Encoder(config, latent_dim=200)
    encoder.alpha = alpha
    encoder.level = level

    # Create a dummy input tensor of the specified shape
    input_tensor = torch.randn(input_shape)

    # Forward pass through the encoder
    mu, logvar = encoder(input_tensor)

    # Check if the output shape is as expected
    assert (
        mu.shape == expected_shape
    ), f"Output shape mismatch for mu at level {level} with alpha {alpha}"
    assert (
        logvar.shape == expected_shape
    ), f"Output shape mismatch for logvar at level {level} with alpha {alpha}"


# Additional tests can be added for other aspects like layer weights, gradients, etc.
