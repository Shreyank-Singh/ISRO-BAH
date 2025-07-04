# validation.py
from tif_processing import TIFProcessor
import numpy as np


def validate_tif_processing():
    processor = TIFProcessor()

    # Test loading
    images, metadata = processor.load_tif_burst('test_tif_burst')
    print(f"Successfully loaded {len(images)} images")

    # Test processing
    result, ref_meta = process_tif_burst('test_tif_burst')
    print(f"Super-resolution result shape: {result.shape}")

    # Test saving
    processor.save_tif_image(result, 'validation_output.tif', ref_meta)
    print("Validation completed successfully!")


if __name__ == "__main__":
    validate_tif_processing()
