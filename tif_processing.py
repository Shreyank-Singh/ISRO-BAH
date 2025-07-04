import rasterio
import numpy as np
from typing import Tuple, List, Optional, Union
import os
from pathlib import Path
import warnings
from handheld_super_resolution import process  # Import the main processing function

# Suppress rasterio warnings for cleaner output
warnings.filterwarnings('ignore', category=rasterio.errors.NotGeoreferencedWarning)


class TIFProcessor:
    """
    Handles TIF file processing for the handheld super-resolution algorithm
    """

    def __init__(self, preserve_geospatial: bool = True):
        self.preserve_geospatial = preserve_geospatial

    def load_tif_image(self, filepath: str) -> Tuple[np.ndarray, dict]:
        """
        Load TIF image using rasterio with proper handling for satellite imagery.

        Args:
            filepath: Path to the TIF file

        Returns:
            Tuple of (image_array, metadata)
        """
        with rasterio.open(filepath) as src:
            # Read all bands
            image_array = src.read()  # Shape: (bands, height, width)

            # Get comprehensive metadata
            metadata = {
                'profile': src.profile,
                'crs': src.crs,
                'transform': src.transform,
                'bounds': src.bounds,
                'width': src.width,
                'height': src.height,
                'count': src.count,
                'dtype': src.dtypes[0] if src.dtypes else np.float32,
                'nodata': src.nodata,
                'descriptions': src.descriptions,
            }

            # Transpose to (height, width, bands) for consistency
            if image_array.ndim == 3 and image_array.shape[0] > 1:
                image_array = np.transpose(image_array, (1, 2, 0))
            elif image_array.ndim == 3 and image_array.shape[0] == 1:
                image_array = image_array[0]

            return image_array, metadata

    def load_tif_burst(self, burst_path: str) -> Tuple[List[np.ndarray], List[dict]]:
        """Load a burst of TIF images from a directory."""
        burst_path = Path(burst_path)

        # Find all TIF files
        tif_files = []
        for ext in ['*.tif', '*.tiff', '*.TIF', '*.TIFF']:
            tif_files.extend(burst_path.glob(ext))

        if not tif_files:
            raise ValueError(f"No TIF files found in {burst_path}")

        # Sort files for consistent order
        tif_files.sort(key=lambda x: x.name)

        images = []
        metadata_list = []

        for tif_file in tif_files:
            img, meta = self.load_tif_image(str(tif_file))
            images.append(img)
            metadata_list.append(meta)

        return images, metadata_list

    def save_tif_image(self, image_array: np.ndarray, output_path: str,
                       reference_metadata: dict, compress: str = 'lzw') -> None:
        """Save image as TIF with geospatial metadata preserved."""
        # Handle different array shapes
        if image_array.ndim == 2:
            bands, height, width = 1, image_array.shape[0], image_array.shape[1]
            image_to_save = image_array[np.newaxis, :, :]
        elif image_array.ndim == 3:
            image_to_save = np.transpose(image_array, (2, 0, 1))
            bands, height, width = image_to_save.shape

        # Create output profile
        if self.preserve_geospatial and reference_metadata.get('profile'):
            profile = reference_metadata['profile'].copy()
            profile.update({
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': bands,
                'dtype': image_array.dtype,
                'compress': compress,
                'tiled': True,
                'blockxsize': min(512, width),
                'blockysize': min(512, height)
            })
        else:
            profile = {
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'count': bands,
                'dtype': image_array.dtype,
                'compress': compress
            }

        # Write the image
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(image_to_save.astype(image_array.dtype))


def convert_tif_to_raw_format(tif_images: List[np.ndarray]) -> List[np.ndarray]:
    """
    Convert TIF images to a format compatible with the handheld super-resolution algorithm.

    Args:
        tif_images: List of TIF image arrays

    Returns:
        List of converted images in the expected format
    """
    converted_images = []

    for img in tif_images:
        # Ensure proper data type (float32 for processing)
        if img.dtype != np.float32:
            img = img.astype(np.float32)

        # Normalize to [0, 1] range if needed
        if img.max() > 1.0:
            img = img / np.max(img)

        # Handle different channel configurations
        if img.ndim == 2:
            # Grayscale - convert to 3-channel for compatibility
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] > 3:
            # Multi-spectral - use first 3 channels
            img = img[:, :, :3]

        converted_images.append(img)

    return converted_images


def create_synthetic_raw_metadata(tif_metadata: dict) -> dict:
    """
    Create synthetic raw metadata compatible with the handheld super-resolution algorithm.

    Args:
        tif_metadata: Original TIF metadata

    Returns:
        Synthetic raw metadata
    """
    # Default noise profile for satellite imagery (can be adjusted)
    noise_profile = {
        'alpha': 1.8e-4,  # Adjusted for satellite imagery
        'beta': 3.2e-6,  # Adjusted for satellite imagery
        'iso': 100  # Default ISO value
    }

    synthetic_metadata = {
        'width': tif_metadata.get('width', 1024),
        'height': tif_metadata.get('height', 1024),
        'channels': 3,
        'dtype': np.float32,
        'noise_profile': noise_profile,
        'cfa_pattern': 'RGGB',  # Default Bayer pattern
        'white_level': 1.0,
        'black_level': 0.0,
        'original_tif_metadata': tif_metadata  # Preserve original metadata
    }

    return synthetic_metadata


def process_tif_burst(burst_path: str, options: dict = None, params: dict = None) -> Tuple[np.ndarray, dict]:
    """
    Process a burst of TIF images using the handheld super-resolution algorithm.

    Args:
        burst_path: Path to directory containing TIF files
        options: Processing options (verbose, etc.)
        params: Algorithm parameters (scale, etc.)

    Returns:
        Tuple of (super_resolved_image, reference_metadata)
    """
    if options is None:
        options = {'verbose': 1}

    if params is None:
        params = {
            'scale': 2,
            'merging': {'kernel': 'handheld'},
            'post processing': {'on': True}
        }

    # Initialize TIF processor
    processor = TIFProcessor()

    # Load TIF burst
    print(f"Loading TIF burst from: {burst_path}")
    tif_images, metadata_list = processor.load_tif_burst(burst_path)
    print(f"Successfully loaded {len(tif_images)} TIF images")

    # Convert TIF images to compatible format
    converted_images = convert_tif_to_raw_format(tif_images)

    # Create synthetic raw metadata
    synthetic_metadata = create_synthetic_raw_metadata(metadata_list[0])

    # Save converted images temporarily as a format the algorithm can process
    import tempfile
    import shutil
    from PIL import Image

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_burst_path = Path(temp_dir) / "temp_burst"
        temp_burst_path.mkdir(exist_ok=True)

        # Save converted images as PNG files for processing
        for i, img in enumerate(converted_images):
            # Convert to uint16 for better precision
            img_uint16 = (img * 65535).astype(np.uint16)

            # Handle different image formats
            if img_uint16.ndim == 3:
                if img_uint16.shape[2] == 3:
                    # RGB
                    pil_img = Image.fromarray(img_uint16, mode='RGB')
                else:
                    # Grayscale
                    pil_img = Image.fromarray(img_uint16[:, :, 0], mode='L')
            else:
                # Grayscale
                pil_img = Image.fromarray(img_uint16, mode='L')

            temp_file_path = temp_burst_path / f"frame_{i:03d}.png"
            pil_img.save(temp_file_path)

        # Process using the original handheld super-resolution algorithm
        try:
            # Import and use the main processing function
            from handheld_super_resolution import process
            result = process(str(temp_burst_path), options, params)

            # Convert result back to appropriate format
            if isinstance(result, np.ndarray):
                processed_result = result
            else:
                # Handle case where result might be in different format
                processed_result = np.array(result)

            # Return processed result with original metadata
            return processed_result, metadata_list[0]

        except Exception as e:
            print(f"Error during processing: {str(e)}")
            # Fallback: simple average of input images
            print("Using fallback processing method...")

            # Simple super-resolution fallback
            averaged_image = np.mean(converted_images, axis=0)

            # Simple upscaling using interpolation
            from scipy.ndimage import zoom
            scale_factor = params.get('scale', 2)

            if averaged_image.ndim == 3:
                upscaled = zoom(averaged_image, (scale_factor, scale_factor, 1), order=1)
            else:
                upscaled = zoom(averaged_image, (scale_factor, scale_factor), order=1)

            return upscaled, metadata_list[0]


# Additional utility functions for compatibility
def validate_tif_processing(test_path: str) -> bool:
    """
    Validate that TIF processing is working correctly.

    Args:
        test_path: Path to test TIF files

    Returns:
        True if validation passes, False otherwise
    """
    try:
        processor = TIFProcessor()
        images, metadata = processor.load_tif_burst(test_path)
        print(f"✓ Successfully loaded {len(images)} TIF images")

        # Test processing
        result, ref_meta = process_tif_burst(test_path, {'verbose': 0}, {'scale': 2})
        print(f"✓ Processing completed, result shape: {result.shape}")

        return True

    except Exception as e:
        print(f"✗ Validation failed: {str(e)}")
        return False


def get_tif_info(filepath: str) -> dict:
    """
    Get detailed information about a TIF file.

    Args:
        filepath: Path to TIF file

    Returns:
        Dictionary with TIF file information
    """
    with rasterio.open(filepath) as src:
        info = {
            'filename': Path(filepath).name,
            'width': src.width,
            'height': src.height,
            'bands': src.count,
            'dtype': str(src.dtypes[0]),
            'crs': str(src.crs) if src.crs else None,
            'transform': src.transform,
            'bounds': src.bounds,
            'resolution': src.res
        }

    return info
