"""Run the included synthetic paired-modality benchmark and write presentation-ready results."""
from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import (RESTORERS, DegradationConfig, degrade, estimate_bpm,
                          make_demo_pair, normalized_correlation, psnr, rms_contrast, ssim_global)

ROOT = Path(__file__).parent
OUT = ROOT / 'outputs'
OUT.mkdir(exist_ok=True)

def panel(clean, bad, restored, title, path):
    imgs = [clean, bad, restored]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, im, name in zip(axes, imgs, ['Clean reference', 'Degraded input', title]):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im, cmap=None if im.ndim == 3 else 'inferno')
        ax.set_title(name); ax.axis('off')
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)

def method_grid(clean, bad, variants, modality, path):
    """One-frame visual comparison of all restoration choices."""
    fig, axes = plt.subplots(1, 2 + len(variants), figsize=(4*(2+len(variants)), 4))
    items = [('Clean reference', clean), ('Degraded input', bad), *variants.items()]
    for ax, (name, im) in zip(axes, items):
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.ndim == 3 else im,
                  cmap=None if im.ndim == 3 else 'inferno')
        ax.set_title(name.replace('_', ' ')); ax.axis('off')
    fig.suptitle(f'{modality.title()} image enhancement: visual comparison', y=1.02, fontsize=15)
    fig.tight_layout(); fig.savefig(path, dpi=160, bbox_inches='tight'); plt.close(fig)

def main():
    optical, thermal, bvp = make_demo_pair()
    cfg = DegradationConfig()
    modalities = {'optical': optical, 'thermal': thermal}
    rows = []
    all_signals = {}
    for modality, frames in modalities.items():
        bad = np.asarray([degrade(f, cfg) for f in frames])
        preview = {}
        for method, fn in RESTORERS.items():
            restored = np.asarray([fn(f) for f in bad])
            preview[method] = restored[70]
            mae = float(np.mean(np.abs(restored.astype(float)-frames.astype(float))))
            # Temporal ROI intensity baseline. (Implemented locally to make ROI explicit.)
            vals=[]
            for f in restored:
                h,w=f.shape[:2]; roi=f[int(.25*h):int(.78*h),int(.28*w):int(.72*w)]
                vals.append(roi[:,:,1].mean() if modality=='optical' and roi.ndim==3 else roi.mean())
            vals=np.asarray(vals)
            all_signals[(modality,method)] = vals
            rows.append({'modality':modality,'method':method,
                         'PSNR_dB':np.mean([psnr(a,b) for a,b in zip(frames,restored)]),
                         'SSIM_global':np.mean([ssim_global(a,b) for a,b in zip(frames,restored)]),
                         'MAE_8bit':mae,
                         'RMS_contrast':np.mean([rms_contrast(f) for f in restored]),
                         'BVP_correlation':normalized_correlation(vals,bvp),
                         'estimated_BPM':estimate_bpm(vals,30)})
            if method == 'gaussian_clahe': panel(frames[70],bad[70],restored[70],method,OUT/f'{modality}_comparison.png')
        method_grid(frames[70], bad[70], preview, modality, OUT/f'{modality}_all_methods.png')
    result = pd.DataFrame(rows)
    result.to_csv(OUT/'metrics.csv',index=False)
    print(result.round(3).to_string(index=False))
    fig, axes=plt.subplots(2,1,figsize=(12,6),sharex=True)
    for ax,(modality,_) in zip(axes,modalities.items()):
        for method in RESTORERS:
            y=all_signals[(modality,method)]; y=(y-y.mean())/(y.std()+1e-8)
            ax.plot(y,label=method,alpha=.85)
        ref=(bvp-bvp.mean())/bvp.std()
        ax.plot(ref,'k--',lw=1.4,label='known synthetic BVP')
        ax.set_title(f'{modality.title()} waveform proxy (central ROI)'); ax.legend(ncol=3,fontsize=8); ax.set_ylabel('z-score')
    axes[-1].set_xlabel('Frame (30 fps)'); fig.tight_layout(); fig.savefig(OUT/'waveform_comparison.png',dpi=180); plt.close(fig)
    best=result.sort_values(['modality','SSIM_global'],ascending=[True,False]).groupby('modality').first().reset_index()
    print('\nBest SSIM methods:\n',best[['modality','method','PSNR_dB','SSIM_global','RMS_contrast','BVP_correlation','estimated_BPM']].round(3).to_string(index=False))

if __name__ == '__main__': main()
