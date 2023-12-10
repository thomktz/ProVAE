import torch
import pytest
from provae import ProVAE


@pytest.fixture
def config():
    # Example configuration, modify as per your actual configuration
    return [
        {"resolution": 4, "channels": 512},
        {"resolution": 8, "channels": 256},
        # Add more configurations as needed
    ]


@pytest.fixture
def latent_dim():
    return 100  # Modify as needed


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def sample_input(config, device):
    # Create a sample input tensor of appropriate size
    input_resolution = config[0]["resolution"]
    return torch.randn(1, 3, input_resolution, input_resolution).to(device)


@pytest.fixture
def provae(latent_dim, config, device):
    return ProVAE(latent_dim, config, device).to(device)


def test_encoder_output(provae, sample_input):
    mu, logvar = provae.encoder(sample_input)
    assert mu.shape == (1, provae.latent_dim)
    assert logvar.shape == (1, provae.latent_dim)


def test_reparameterize(provae, sample_input):
    mu, logvar = provae.encoder(sample_input)
    z = provae.reparameterize(mu, logvar)
    assert z.shape == mu.shape


def test_decoder_output(provae, sample_input):
    mu, logvar = provae.encoder(sample_input)
    z = provae.reparameterize(mu, logvar)
    reconstructed = provae.decoder(z)
    assert reconstructed.shape == sample_input.shape


def test_provae_forward_pass(provae, sample_input):
    reconstructed, mu, logvar = provae(sample_input)
    assert reconstructed.shape == sample_input.shape
    assert mu.shape == (1, provae.latent_dim)
    assert logvar.shape == (1, provae.latent_dim)
