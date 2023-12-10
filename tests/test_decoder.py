import pytest
import torch
from provae.provae.decoder import Decoder
from provae.provae.config import config


@pytest.mark.parametrize(
    "level, alpha, input_shape, expected_shape",
    [
        (0, 1, (1, 200), (1, 3, 4, 4)),  # Level 0, no blending
        (1, 0.5, (1, 200), (1, 3, 8, 8)),  # Level 1, blending
        (1, 1, (1, 200), (1, 3, 8, 8)),  # Level 1, no blending
        (2, 0.5, (1, 200), (1, 3, 16, 16)),  # Level 2, blending
    ],
)
def test_decoder_output_shape(level, alpha, input_shape, expected_shape):
    decoder = Decoder(config, latent_dim=200)
    decoder.level = level
    decoder.alpha = alpha
    input_tensor = torch.randn(input_shape)
    output = decoder(input_tensor)
    assert (
        output.shape == expected_shape
    ), f"Decoder output shape mismatch at level {level} with alpha {alpha}"
