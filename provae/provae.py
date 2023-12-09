# Make it a Config class?
example_config = [
    {
        "resolution": 4,
        "transition_epochs": 0,
        "stabilization_epochs": 10,
        "batch_size": 128,
    },
    {
        "resolution": 8,
        "transition_epochs": 10,
        "stabilization_epochs": 10,
        "batch_size": 128,
    },
]


class ProVAE:
    """
    Progressive-Growing Variational Autoencoder.

    Parameters
    ----------
    latent_dim : int
        The dimensionality of the latent space.
    config : list of dicts
        The configuration for the progressive growing.
    device : str
        The device to run the model on.
    """

    def __init__(self, latent_dim, config, device):
        self.latent_dim = latent_dim
        self.config = config
        self.device = device
