import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

class CTNoisyTransform:

    def __init__(self, 
                 apply_blur=True, blur_kernel_size=5, blur_sigma=1.0,
                 apply_gaussian_noise=False, gaussian_mean=0.0, gaussian_std=0.05,
                 apply_poisson_noise=False, poisson_scale=10.0):
        self.apply_blurring = apply_blur
        self.blur_kernel_size = blur_kernel_size
        self.blur_sigma = blur_sigma
        
        self.apply_gaussian_noise = apply_gaussian_noise
        self.gaussian_mean = gaussian_mean
        self.gaussian_std = gaussian_std
        
        self.apply_poisson_noise = apply_poisson_noise
        self.poisson_scale = poisson_scale

    @staticmethod
    def get_gaussian_kernel(kernel_size=5, sigma=1.0):
        ax = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    @classmethod
    def apply_blur(cls, tensor, kernel_size=5, sigma=1.0):
        kernel = cls.get_gaussian_kernel(kernel_size, sigma)
        C = tensor.shape[0]
        kernel = kernel.expand(C, 1, kernel_size, kernel_size)
        tensor = tensor.unsqueeze(0)
        padding = kernel_size // 2
        blurred = F.conv2d(tensor, kernel, padding=padding, groups=C)
        return blurred.squeeze(0)

    @staticmethod
    def add_gaussian_noise(tensor, mean=0.0, std=0.05):
        noise = torch.randn(tensor.size()) * std + mean
        return tensor + noise

    @staticmethod
    def add_poisson_noise(tensor, scale=10.0):
        noisy = torch.poisson(tensor * scale) / scale
        return noisy

    def __call__(self, tensor):
        if self.apply_blurring:
            tensor = CTNoisyTransform.apply_blur(tensor, 
                                                   kernel_size=self.blur_kernel_size, 
                                                   sigma=self.blur_sigma)
        if self.apply_gaussian_noise:
            tensor = CTNoisyTransform.add_gaussian_noise(tensor, 
                                                         mean=self.gaussian_mean, 
                                                         std=self.gaussian_std)
        if self.apply_poisson_noise:
            tensor = CTNoisyTransform.add_poisson_noise(tensor, scale=self.poisson_scale)
        tensor = torch.clamp(tensor, 0.0, 1.0)
        
        return tensor

# --- Example Usage ---
# Assume CT images are already loaded as a tensor with shape (C, H, W),
# for instance, a 64x64 CT image.
# Here is how you can integrate this class in a transformation pipeline:

# Compose a pipeline that converts the input to a tensor (if needed) and applies the CT noise transforms.
transform_pipeline = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),  # Convert input to a tensor if necessary.
    CTNoisyTransform(apply_blur=True, blur_kernel_size=5, blur_sigma=1.0,
                     apply_gaussian_noise=True, gaussian_std=0.05)
    # To add Poisson noise instead, set `apply_poisson_noise=True` and adjust parameters.
])

# Example integration with a dataset:
# dataset = pytorch_dataset(split='train', transform=transform_pipeline)
# img, target = dataset[0]
