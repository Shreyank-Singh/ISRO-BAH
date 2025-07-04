import argparse
from pathlib import Path
import numpy as np
import sys
import os


def main():
    parser = argparse.ArgumentParser(description='Handheld Super-Resolution with TIF support')
    parser.add_argument('--impath', type=str, required=True, help='Path to input images')
    parser.add_argument('--outpath', type=str, required=True, help='Output path')
    parser.add_argument('--format', type=str, choices=['auto', 'dng', 'tif'],
                        default='auto', help='Input format')
    parser.add_argument('--scale', type=float, default=2.0, help='Super-resolution scale')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')

    args = parser.parse_args()

    # Ensure input path exists
    if not Path(args.impath).exists():
        print(f"Error: Input path '{args.impath}' does not exist")
        sys.exit(1)

    # Auto-detect format
    if args.format == 'auto':
        input_path = Path(args.impath)
        if input_path.is_dir():
            tif_files = list(input_path.glob('*.tif')) + list(input_path.glob('*.tiff'))
            dng_files = list(input_path.glob('*.dng')) + list(input_path.glob('*.DNG'))

            if tif_files:
                format_type = 'tif'
                print(f"Auto-detected TIF format ({len(tif_files)} files found)")
            elif dng_files:
                format_type = 'dng'
                print(f"Auto-detected DNG format ({len(dng_files)} files found)")
            else:
                print("Error: No TIF or DNG files found in the specified directory")
                sys.exit(1)
        else:
            format_type = 'tif' if args.impath.lower().endswith(('.tif', '.tiff')) else 'dng'
    else:
        format_type = args.format

    # Set up processing parameters
    options = {'verbose': args.verbose}
    params = {
        'scale': args.scale,
        'merging': {'kernel': 'handheld'},
        'post processing': {'on': True}
    }

    # Process based on format
    if format_type == 'tif':
        print("Processing TIF files...")

        try:
            from tif_processing import TIFProcessor, process_tif_burst

            # Process TIF burst
            result, metadata = process_tif_burst(
                args.impath,
                options=options,
                params=params
            )

            print(f"Processing completed. Result shape: {result.shape}")

            # Save result
            processor = TIFProcessor()
            output_path = Path(args.outpath)

            if output_path.suffix.lower() in ['.tif', '.tiff']:
                # Save as TIF with metadata
                processor.save_tif_image(result, str(output_path), metadata)
                print(f"Result saved as TIF: {output_path}")
            else:
                # Save as PNG/JPG for visualization
                from PIL import Image

                # Convert to uint8 for standard image formats
                if result.dtype != np.uint8:
                    if result.max() <= 1.0:
                        result_uint8 = (result * 255).astype(np.uint8)
                    else:
                        result_uint8 = np.clip(result, 0, 255).astype(np.uint8)
                else:
                    result_uint8 = result

                # Handle different image shapes
                if result_uint8.ndim == 3:
                    if result_uint8.shape[2] == 3:
                        Image.fromarray(result_uint8, mode='RGB').save(output_path)
                    else:
                        Image.fromarray(result_uint8[:, :, 0], mode='L').save(output_path)
                else:
                    Image.fromarray(result_uint8, mode='L').save(output_path)

                print(f"Result saved as image: {output_path}")

        except ImportError as e:
            print(f"Error importing TIF processing modules: {e}")
            print("Make sure rasterio is installed: pip install rasterio")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing TIF files: {e}")
            sys.exit(1)

    else:
        print("Processing DNG files...")

        try:
            # Use original DNG processing
            from handheld_super_resolution import process
            result = process(args.impath, options, params)

            # Save result
            if args.outpath.lower().endswith(('.png', '.jpg', '.jpeg')):
                from PIL import Image
                if result.dtype != np.uint8:
                    result = (result * 255).astype(np.uint8)
                Image.fromarray(result).save(args.outpath)
            else:
                # Save as DNG or other format
                # This would require additional DNG processing
                print("DNG output not implemented in this version")

        except ImportError as e:
            print(f"Error importing handheld super-resolution: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"Error processing DNG files: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
