import argparse
from pathlib import Path
import numpy as np
import sys
import os

def find_image_pairs(input_dir):
    """
    Finds all unique base names that have both _0.tif and _1.tif (or .tiff) in the directory.
    Returns a list of (base_name, [file0, file1]) tuples.
    """
    input_dir = Path(input_dir)
    files = list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff"))
    pair_dict = {}
    for f in files:
        name = f.stem
        if name.endswith("_0") or name.endswith("_1"):
            base = name[:-2]
            suffix = name[-2:]
            if base not in pair_dict:
                pair_dict[base] = {}
            pair_dict[base][suffix] = f
    # Only keep pairs that have both _0 and _1
    pairs = []
    for base, d in pair_dict.items():
        if "_0" in d and "_1" in d:
            pairs.append( (base, [d["_0"], d["_1"]]) )
    return pairs

def main():
    parser = argparse.ArgumentParser(description='Batch Pairwise TIF Super-Resolution')
    parser.add_argument('--impath', type=str, required=True, help='Path to input images')
    parser.add_argument('--outdir', type=str, default="results", help='Directory to save results')
    parser.add_argument('--scale', type=float, default=2.0, help='Super-resolution scale')
    parser.add_argument('--verbose', type=int, default=1, help='Verbosity level')
    args = parser.parse_args()

    input_dir = Path(args.impath)
    outdir = Path(args.outdir)
    outdir.mkdir(exist_ok=True)

    # Find pairs
    pairs = find_image_pairs(input_dir)
    if not pairs:
        print("No valid image pairs found! Make sure you have files like name_0.tif and name_1.tif")
        sys.exit(1)

    from tif_processing import TIFProcessor, process_tif_burst

    for base, (img0, img1) in pairs:
        print(f"Processing pair: {base} ({img0.name}, {img1.name})")
        # Create a temp directory for this pair
        import tempfile, shutil
        with tempfile.TemporaryDirectory() as tempdir:
            tempdir = Path(tempdir)
            # Copy both images into tempdir
            temp_img0 = tempdir / img0.name
            temp_img1 = tempdir / img1.name
            shutil.copy(str(img0), str(temp_img0))
            shutil.copy(str(img1), str(temp_img1))
            # Process this pair as a burst
            result, metadata = process_tif_burst(str(tempdir), options={'verbose': args.verbose}, params={'scale': args.scale})
            # Save output
            outname = f"{base}_AP.tif"
            outpath = outdir / outname
            processor = TIFProcessor()
            processor.save_tif_image(result, str(outpath), metadata)
            print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
