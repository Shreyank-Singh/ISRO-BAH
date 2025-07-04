import os
import glob
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from skimage.transform import resize
from scipy.stats import spearmanr, pearsonr

# Import no-reference IQA metrics
try:
    from piq import brisque, niqe, piqe  # pip install piq
except ImportError:
    brisque = niqe = piqe = None

try:
    import torch
    from torchvision import transforms
    from torchvision.models import resnet18, ResNet18_Weights
except ImportError:
    torch = None

def compute_full_reference(sr_img, gt_img):
    sr = img_as_float(sr_img)
    gt = img_as_float(gt_img)
    # Resize ground-truth if needed
    if sr.shape != gt.shape:
        print(f"Resizing ground-truth from {gt.shape} to {sr.shape}")
        if sr.ndim == 2:
            gt = resize(gt, sr.shape, order=3, mode='reflect', anti_aliasing=True)
        else:
            gt = resize(gt, sr.shape, order=3, mode='reflect', anti_aliasing=True, channel_axis=-1)
    # Handle grayscale and color images
    if sr.ndim == 2 or gt.ndim == 2:
        multichannel = False
    else:
        multichannel = True
    return {
        'PSNR': psnr(gt, sr, data_range=1.0),
        'SSIM': ssim(gt, sr, multichannel=multichannel, data_range=1.0),
        'RMSE': np.sqrt(mse(gt, sr))
    }

def compute_blind_metrics(img):
    results = {}
    if brisque:
        try:
            results['BRISQUE'] = float(brisque(img))
        except Exception:
            results['BRISQUE'] = None
    if niqe:
        try:
            results['NIQE'] = float(niqe(img))
        except Exception:
            results['NIQE'] = None
    if piqe:
        try:
            results['PIQE'] = float(piqe(img))
        except Exception:
            results['PIQE'] = None
    return results

def compute_deep_feature_score(img, model=None):
    if torch is None:
        return None
    if model is None:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    # Convert image to uint8 if needed
    if img.dtype != np.uint8:
        img = (img / img.max() * 255).astype(np.uint8)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    elif img.shape[2] > 3:
        img = img[:, :, :3]
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        features = model(x)
        score = features.norm().item()
    return score

def strip_ap_suffix(filename):
    base, ext = os.path.splitext(filename)
    if base.endswith('_AP'):
        base = base[:-3]
    return base + ext

def evaluate_folder(sr_dir, gt_dir=None):
    sr_files = sorted(glob.glob(os.path.join(sr_dir, "*.tif")))
    results = []
    for sr_path in sr_files:
        sr_name = os.path.basename(sr_path)
        entry = {'file': sr_name}
        sr_img = io.imread(sr_path)
        gt_img = None
        if gt_dir is not None:
            gt_name = strip_ap_suffix(sr_name)
            gt_path = os.path.join(gt_dir, gt_name)
            if os.path.exists(gt_path):
                gt_img = io.imread(gt_path)
                entry.update(compute_full_reference(sr_img, gt_img))
            else:
                entry['warning'] = f"Ground-truth not found for {sr_name} (looked for {gt_name})"
        entry.update(compute_blind_metrics(sr_img))
        entry['DeepFeature'] = compute_deep_feature_score(sr_img)
        results.append(entry)
    return results

def correlation_analysis(results):
    metrics = ['BRISQUE', 'NIQE', 'PIQE', 'DeepFeature']
    gt_metrics = ['PSNR', 'SSIM', 'RMSE']
    for m in metrics:
        for gt in gt_metrics:
            x = [r[m] for r in results if m in r and gt in r and r[m] is not None and r[gt] is not None]
            y = [r[gt] for r in results if m in r and gt in r and r[m] is not None and r[gt] is not None]
            if len(x) > 2:
                try:
                    spearman = spearmanr(x, y)[0]
                    pearson = pearsonr(x, y)[0]
                    print(f"{m} vs {gt}: Spearman={spearman:.3f}, Pearson={pearson:.3f}")
                except Exception:
                    print(f"{m} vs {gt}: Correlation calculation failed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Quality Assessment for Super-Resolution")
    parser.add_argument('--sr_dir', required=True, help="Directory with super-resolved images")
    parser.add_argument('--gt_dir', default=None, help="Directory with ground-truth images (optional)")
    args = parser.parse_args()
    try:
        results = evaluate_folder(args.sr_dir, args.gt_dir)
        for r in results:
            print(r)
        correlation_analysis(results)
    except Exception as e:
        print("Error during quality assessment:", e)
