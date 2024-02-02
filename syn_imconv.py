import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

def create_object_field(nx, ny, wavelength, z):
    k = 2 * np.pi / wavelength
    phase = np.exp(1j * k * z)
    # Create a simpe object field
    field = np.zeros((nx, ny), dtype=np.complex64)
    field[nx//2-5:nx//2+5, ny//2-5:ny//2+5] = 1.0  # Point source
    field = field * phase
    return field

def create_kernel(nx, ny, pixel_size, wavelength, z):
    k = 2 * np.pi / wavelength
    x = np.linspace(-nx//2, nx//2, nx) * pixel_size
    y = np.linspace(-ny//2, ny//2, ny) * pixel_size
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2 + z**2)
    r[r==0] = 1e-10  # Avoid division by zero
    kernel = np.exp(1j * k * r) / (1j * wavelength * r) * (z / r)
    return kernel.astype(np.complex64)

def generate_hologram(field, kernel, padding_size):
    # Convert to PyTorch tensors
    field_torch = torch.from_numpy(field[np.newaxis, np.newaxis, ...])
    kernel_torch = torch.from_numpy(kernel[np.newaxis, np.newaxis, ...])

    # Add zero padding to the field
    field_torch_padded = F.pad(field_torch, [padding_size] * 4)

    # Complex convolution using PyTorch
    real_conv = F.conv2d(field_torch_padded.real, kernel_torch.real) - F.conv2d(field_torch_padded.imag, kernel_torch.imag)
    imag_conv = F.conv2d(field_torch_padded.real, kernel_torch.imag) + F.conv2d(field_torch_padded.imag, kernel_torch.real)
    hologram = real_conv + 1j * imag_conv

    start = padding_size
    end = -padding_size if padding_size != 0 else None
    hologram_cropped = hologram[0, 0, start:end, start:end].detach().numpy()
    # hologram_cropped /= np.max(np.abs(hologram_cropped))  # Normalize the kernel

    return hologram_cropped


# Parameters
nx, ny = 256, 256  # Image size
pixel_size = 10e-6  # Pixel size in meters
wavelength = 500e-9  # Wavelength of light in meters
z = 0.1  # Distance from object plane to hologram plane in meters
padding_size = 50  # Padding size for zero padding

# Create object field and kernel
object_field = create_object_field(nx, ny, wavelength, z)
kernel = create_kernel(nx, ny, pixel_size, wavelength, z)

# Visualize the object field and the kernel
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(object_field.real, cmap='gray')
plt.title('Object Field')
plt.colorbar()

plt.subplot(1, 2, 2)
plt.imshow(kernel.real, cmap='gray')
plt.title('Real Part of Kernel')
plt.colorbar()
plt.show()

# Generate hologram
hologram = generate_hologram(object_field, kernel, padding_size)

# Display the hologram
plt.figure(figsize=(6, 6))
plt.imshow(hologram.real, cmap='gray')
plt.title('Generated Hologram (real)')
plt.colorbar()
plt.show()

# Display the hologram
plt.figure(figsize=(6, 6))
plt.imshow(hologram.imag, cmap='gray')
plt.title('Generated Hologram (imaginary)')
plt.colorbar()
plt.show()

