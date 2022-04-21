import torch
import torch.nn as nn
from torchvision.transforms.functional import gaussian_blur

class AddGaussianNoise(nn.Module):
    def __init__(self, mean, std, fraction):
        """
        Additive Gaussian noise.

        [TODO:description]

        Parameters
        ----------
        mean : float
            Mean of the Gaussian noise distribution.
        std : float
            Standard deviation of the Gaussian noise distribution.
        fraction : float
            Relative intensity of the noise. Values from 0 to 1.
        """
        super().__init__()
        self.std = std
        self.mean = mean
        self.fraction = fraction 

    def forward(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        noisy = (1 - self.fraction)*tensor + self.fraction*noise
        return noisy

class AddRandomGaussianNoise(nn.Module):
    def __init__(self, mean, std, fraction, distribution='normal'):
        """
        Additive Gaussian noise with random standard deviation.

        [TODO:description]

        Parameters
        ----------
        mean : float
            Mean of the Gaussian noise distribution.
        std : tuple of float
            Range of kernel standard deviations if uniform distribution. Mean
            and standard deviation if normal distribution.
        fraction : float
            Relative intensity of the noise. Values from 0 to 1.
        distribution : float
            'normal' (default) or 'uniform'. 

        Raises
        ------
        ValueError:
            If distribution is not 'normal' or 'uniform'.
        """
        super().__init__()
        self.std = std
        self.mean = mean
        self.fraction = fraction 
        self.distribution = distribution

    def forward(self, tensor):
        if self.distribution == 'normal':
            std = torch.empty(1).normal_(*self.std)
            std = nn.functional.relu(std).item()
        elif self.distribution == 'uniform':
            std = torch.empty(1).uniform_(*self.std).item()
        else:
            raise ValueError('Distribution should be either uniform or normal')
        noise = torch.randn(tensor.size()) * std + self.mean
        noisy = (1 - self.fraction)*tensor + self.fraction*noise
        return noisy

class AddScatter(nn.Module):
    def __init__(self, kernel_size, std, sf):
        """
        Simulates scatter with a Gaussian kernel convolution.
        
        [TODO:description]

        Parameters
        ----------
        kernel_size : float
            Size of the convolution kernel.
        std : float
            Standard deviation of the convolution kernel.
        sf : float
            Scatter fraction.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.std = std
        self.sf = sf

    def forward(self, tensor):
        scatter = gaussian_blur(tensor, self.kernel_size, self.std)
        noisy = (1-self.sf)*tensor + self.sf*scatter
        return noisy

class AddRandomScatter(nn.Module):
    def __init__(self, kernel_size, std, sf, distribution='normal'):
        """
        Simulate scatter with a Gaussian kernel convolution using random 
        parameters.

        The standard deviation and scatter fraction of the kernel are sampled
        from a normal or uniform distribution.

        Parameters
        ----------
        kernel_size : int
            Size of the convolution kernel
        std : tuple of float
            Range of kernel standard deviations if uniform distribution. Mean
            and standard deviation if normal distribution.
        sf : tuple of float
            Range of kernel scatter fractions if uniform distribution. Mean
            and standard deviation if normal distribution.
        distribution : string
            'normal' (default) or 'uniform'.
        
        Raises
        ------
        ValueError:
            If distribution is not 'normal' or 'uniform'.
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.std = std
        self.sf = sf
        self.distribution = distribution

    def forward(self, tensor):
        if self.distribution == 'normal':
            std = torch.empty(1).normal_(*self.std)
            std = nn.functional.relu(std).item()
            sf = torch.empty(1).normal_(*self.sf)
            sf = nn.functional.relu(sf).item()
        elif self.distribution == 'uniform':
            std = torch.empty(1).uniform_(*self.std).item()
            sf = torch.empty(1).uniform_(*self.sf).item()
        else:
            raise ValueError('Distribution should be either uniform or normal')
        
        scatter = gaussian_blur(tensor, self.kernel_size, std)
        noisy = (1-sf)*tensor + sf*scatter
        return noisy
