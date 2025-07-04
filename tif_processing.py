import rasterio
import numpy as np
from typing import Tuple, List, Optional, Union
import os
from pathlib import Path
import warnings

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
