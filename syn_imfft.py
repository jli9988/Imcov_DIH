import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift

def create_object_field(nx, ny, depth_range):
    # Create a simple object field (e.g., a point source or a simple shape)
    # For simplicity, we'll create a point source in the center
    field = np.zeros((nx, ny), dtype=complex)
    field[nx//2-2:nx//2+2, ny//2-2:ny//2+2] = 1  # Point source
    return field

def create_kernel(nx, ny, pixel_size, wavelength, z):
    k = 2 * np.pi / wavelength
    x = np.linspace(-nx/2, nx/2, nx) * pixel_size
    y = np.linspace(-ny/2, ny/2, ny) * pixel_size
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2 + z**2)
    r[r==0] = 1e-10  # Avoid division by zero
    kernel = np.exp(-1j * k * r) / (1j * wavelength * r) * (z / r)
    return kernel


def create_kernel_fft(nx, ny, pixel_size, wavelength, z):
    k = 2 * np.pi / wavelength
    x = np.linspace(-nx//2, nx//2, nx) * pixel_size
    y = np.linspace(-ny//2, ny//2, ny) * pixel_size
    fx, fy = np.meshgrid(1 / x, 1 / y)
    r = np.sqrt(1 - fx**2 * wavelength**2 - fy**2 * wavelength**2)
    r[r==0] = 1e-10  # Avoid division by zero
    kernel_fft = np.exp(-1j * k * r * z)
    phase = np.exp(1j * k * z)
    return kernel_fft, phase


def generate_hologram(field, kernel_fft, phase):
    # Perform convolution using FFT for efficiency
    
    field_fft = fft2(field * phase)

    plt.imshow(np.abs(field_fft), cmap='gray')
    plt.colorbar()
    plt.title('Generated Hologram')
    plt.show()
    # kernel_fft = fft2(fftshift(kernel))  # Kernel needs to be shifted for FFT
    hologram_fft = field_fft * kernel_fft
    hologram = ifft2(hologram_fft)
    return np.abs(hologram)**2  # Returning the intensity

# Parameters
nx, ny = 256, 256  # Image size
pixel_size = 10e-6  # Pixel size in meters
wavelength = 500e-9  # Wavelength of light in meters
z = 0.01  # Distance from object plane to hologram plane in meters

# Create object field and kernel
object_field = create_object_field(nx, ny, depth_range=(0, z))
kernel_fft, phase = create_kernel_fft(nx, ny, pixel_size, wavelength, z)

# Generate hologram
hologram = generate_hologram(object_field, kernel_fft, phase)

# Display the hologram
plt.imshow(hologram, cmap='gray')
plt.colorbar()
plt.title('Generated Hologram')
plt.show()




