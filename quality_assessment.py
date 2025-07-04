import os
import glob
import numpy as np
from skimage import io, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim, mean_squared_error as mse
from scipy.stats import spearmanr, pearsonr

# Import no-reference IQA metrics
try:
    from piq import brisque, niqe, piqe  # pip install piq
except ImportError:
    brisque = niqe = piqe = None

try:
    import torch
    from torchvision import transforms, models
except ImportError:
    torch = None

def compute_full_reference(sr_img, gt_img):
    sr = img_as_float(sr_img)
    gt = img_as_float(gt_img)
    return {
        'PSNR': psnr(gt, sr, data_range=1.0),
        'SSIM': ssim(gt, sr, multichannel=True, data_range=1.0),
        'RMSE': np.sqrt(mse(gt, sr))
    }

def compute_blind_metrics(img):
    results = {}
    if brisque:
        results['BRISQUE'] = brisque(img)
    if niqe:
        results['NIQE'] = niqe(img)
    if piqe:
        results['PIQE'] = piqe(img)
    return results

def compute_deep_feature_score(img, model=None):
    if torch is None:
        return None
    if model is None:
        model = models.resnet18(pretrained=True)
        model.eval()
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    with torch.no_grad():
        x = preprocess(img).unsqueeze(0)
        features = model(x)
        score = features.norm().item()
    return score

def evaluate_folder(sr_dir, gt_dir=None):
    sr_files = sorted(glob.glob(os.path.join(sr_dir, "*.tif")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.tif"))) if gt_dir else [None]*len(sr_files)
    results = []
    for sr_path, gt_path in zip(sr_files, gt_files):
        sr_img = io.imread(sr_path)
        gt_img = io.imread(gt_path) if gt_path else None
        entry = {'file': os.path.basename(sr_path)}
        if gt_img is not None:
            entry.update(compute_full_reference(sr_img, gt_img))
        entry.update(compute_blind_metrics(sr_img))
        entry['DeepFeature'] = compute_deep_feature_score(sr_img)
        results.append(entry)
    return results

def correlation_analysis(results):
    # Correlate blind metrics with PSNR/SSIM if available
    metrics = ['BRISQUE', 'NIQE', 'PIQE', 'DeepFeature']
    gt_metrics = ['PSNR', 'SSIM', 'RMSE']
    for m in metrics:
        for gt in gt_metrics:
            x = [r[m] for r in results if m in r and gt in r]
            y = [r[gt] for r in results if m in r and gt in r]
            if len(x) > 2:
                print(f"{m} vs {gt}: Spearman={spearmanr(x, y)[0]:.3f}, Pearson={pearsonr(x, y)[0]:.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Image Quality Assessment for Super-Resolution")
    parser.add_argument('--sr_dir', required=True, help="Directory with super-resolved images")
    parser.add_argument('--gt_dir', default=None, help="Directory with ground-truth images (optional)")
    args = parser.parse_args()
    results = evaluate_folder(args.sr_dir, args.gt_dir)
    for r in results:
        print(r)
    correlation_analysis(results)
