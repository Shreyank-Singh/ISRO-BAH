from tif_processing import validate_tif_processing, get_tif_info
from pathlib import Path


def test_implementation():
    # Test with your TIF files
    test_path = "path/to/your/tif/files"  # Replace with your actual path

    if Path(test_path).exists():
        print("Testing TIF processing implementation...")

        # Get info about TIF files
        tif_files = list(Path(test_path).glob("*.tif")) + list(Path(test_path).glob("*.tiff"))

        if tif_files:
            print(f"Found {len(tif_files)} TIF files:")
            for tif_file in tif_files[:3]:  # Show first 3 files
                info = get_tif_info(str(tif_file))
                print(f"  {info['filename']}: {info['width']}x{info['height']}, {info['bands']} bands")

        # Validate processing
        if validate_tif_processing(test_path):
            print("✓ TIF processing validation passed!")
        else:
            print("✗ TIF processing validation failed!")
    else:
        print(f"Test path '{test_path}' does not exist")


if __name__ == "__main__":
    test_implementation()
