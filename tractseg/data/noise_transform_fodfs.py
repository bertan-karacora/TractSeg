from typing import Tuple

import bonndit as bd
import numpy as np

from batchgenerators.transforms.abstract_transforms import AbstractTransform


def augment_gaussian_noise(
    data_sample: np.ndarray,
    noise_variance: Tuple[float, float] = (0, 0.1),
    p_per_channel: float = 1,
    per_channel: bool = False,
) -> np.ndarray:
    order = bd.utils.tensor.get_order(data_sample[:, 0, 0])
    multiplicities = np.asarray(bd.utils.tensor.MULTIPLIER[order])

    variance = None
    if not per_channel:
        variance = noise_variance[0] if noise_variance[0] == noise_variance[1] else np.random.uniform(noise_variance[0], noise_variance[1])

    for c in range(data_sample.shape[0]):
        if np.random.uniform() < p_per_channel:
            variance_here = (
                variance
                if variance is not None
                else noise_variance[0] if noise_variance[0] == noise_variance[1] else np.random.uniform(noise_variance[0], noise_variance[1])
            )

            variance_here /= multiplicities[c]

            data_sample[c] = data_sample[c] + np.random.normal(0.0, variance_here, size=data_sample[c].shape)
    return data_sample


class GaussianNoiseTransformFodfs(AbstractTransform):
    # How this is called can be done much better but this is necessary
    # to keep the same call signature as batchgenerators' noise transform API.
    def __init__(
        self,
        noise_variance=(0, 0.1),
        p_per_sample=1,
        p_per_channel: float = 1,
        per_channel: bool = False,
        data_key="data",
    ):
        """
        Adds additive Gaussian Noise

        :param noise_variance: variance is uniformly sampled from that range
        :param p_per_sample:
        :param p_per_channel:
        :param per_channel: if True, each channel will get its own variance sampled from noise_variance
        :param data_key:

        CAREFUL: This transform will modify the value range of your data!
        """
        self.p_per_sample = p_per_sample
        self.data_key = data_key
        self.noise_variance = noise_variance
        self.p_per_channel = p_per_channel
        self.per_channel = per_channel

    def __call__(self, **data_dict):
        for b in range(len(data_dict[self.data_key])):
            if np.random.uniform() < self.p_per_sample:
                data_dict[self.data_key][b] = augment_gaussian_noise(
                    data_dict[self.data_key][b],
                    self.noise_variance,
                    self.p_per_channel,
                    self.per_channel,
                )
        return data_dict
