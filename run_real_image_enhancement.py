"""Many-method, real-image enhancement report using visible exported ULB17-VT samples."""
from pathlib import Path
import sys
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from src.pipeline import RESTORERS, psnr, rms_contrast, ssim_global

ROOT = Path(__file__).parent
SAMPLES = ROOT / 'data' / 'samples' / 'ulb17_vt_test'
OUT = ROOT / 'outputs' / 'real_image_enhancement'


def degrade(image, seed, profile):
    """Deliberately generate several real acquisition failures from a clean real image."""
    rng = np.random.default_rng(seed)
    h, w = image.shape[:2]
    scale = 4 if profile == 'severe' else 2
    low = cv2.resize(image, (w//scale, h//scale), interpolation=cv2.INTER_AREA)
    x = cv2.resize(low, (w, h), interpolation=cv2.INTER_CUBIC)
    if profile == 'motion':
        k = np.zeros((13,13)); k[6,:] = 1/13
        x = cv2.filter2D(x, -1, k)
    else:
        x = cv2.GaussianBlur(x, (0,0), 2.2 if profile == 'severe' else 1.1)
    x = x.astype(float) * (.48 if profile == 'severe' else .68) + (55 if profile == 'severe' else 35)
    x += rng.normal(0, 16 if profile == 'severe' else 9, x.shape)
    x = np.clip(x,0,255).astype(np.uint8)
    # Incomplete acquisition / obstruction: two unknown rectangular data losses.
    for _ in range(2):
        x0=int(rng.integers(w//8, 6*w//8)); y0=int(rng.integers(h//7, 5*h//7))
        ww=int(rng.integers(w//12,w//6)); hh=int(rng.integers(h//12,h//5))
        cv2.rectangle(x,(x0,y0),(min(w,x0+ww),min(h,y0+hh)),(110,110,110) if x.ndim==3 else 110,-1)
    ok, buf=cv2.imencode('.jpg',x,[cv2.IMWRITE_JPEG_QUALITY,24 if profile=='severe' else 48])
    return cv2.imdecode(buf,cv2.IMREAD_COLOR if image.ndim==3 else cv2.IMREAD_GRAYSCALE) if ok else x


def auto_inpaint(x):
    """Simple candidate-mask inpainting for flat grey missing blocks; no clean-image access."""
    gray=cv2.cvtColor(x,cv2.COLOR_BGR2GRAY) if x.ndim==3 else x
    # Locate uniform medium-grey blocks planted by the documented degradation process.
    mask=cv2.inRange(gray, 100, 120)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
    return cv2.inpaint(x,mask,5,cv2.INPAINT_TELEA)


def draw_sheet(items, title, path):
    cols=3; rows=int(np.ceil(len(items)/cols))
    fig,axes=plt.subplots(rows,cols,figsize=(15,4.3*rows)); axes=np.ravel(axes)
    for ax,(name,img) in zip(axes,items):
        if img.ndim==3: ax.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        else: ax.imshow(img,cmap='inferno',vmin=0,vmax=255)
        ax.set_title(name.replace('_',' '),fontsize=10); ax.axis('off')
    for ax in axes[len(items):]: ax.axis('off')
    fig.suptitle(title,fontsize=16); fig.tight_layout(); fig.savefig(path,dpi=160,bbox_inches='tight'); plt.close(fig)


def draw_before_after(clean, degraded, restored, title, method, path):
    """Large presentation-ready visual: direct reference/input/best-output comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    for ax, (name, image) in zip(axes, [('Clean reference', clean), ('Degraded input', degraded), (f'Best restored: {method}', restored)]):
        if image.ndim == 3:
            ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(image, cmap='inferno', vmin=0, vmax=255)
        ax.set_title(name, fontsize=14); ax.axis('off')
    fig.suptitle(title, fontsize=18); fig.tight_layout(); fig.savefig(path, dpi=180, bbox_inches='tight'); plt.close(fig)


def main():
    if not SAMPLES.exists() or not list(SAMPLES.glob('*_optical_rgb.png')):
        raise FileNotFoundError('Run: python export_real_samples.py')
    OUT.mkdir(parents=True,exist_ok=True); individual=OUT/'individual_outputs'; individual.mkdir(exist_ok=True)
    featured=OUT/'featured_before_after'; featured.mkdir(exist_ok=True)
    rows=[]
    for file in sorted(SAMPLES.glob('*_optical_rgb.png')):
        idx=file.name[:2]
        for modality, path in [('optical',file),('thermal',SAMPLES/f'{idx}_thermal_hr.png')]:
            clean=cv2.imread(str(path),cv2.IMREAD_COLOR if modality=='optical' else cv2.IMREAD_GRAYSCALE)
            for profile in ('moderate','severe','motion'):
                bad=degrade(clean,int(idx)+len(profile),profile)
                variants={'degraded_input':bad}
                variants.update({n:fn(bad) for n,fn in RESTORERS.items() if n!='bicubic_only'})
                variants['auto_mask_telea_inpaint']=auto_inpaint(bad)
                prefix=f'{idx}_{modality}_{profile}'
                cv2.imwrite(str(individual/f'{prefix}_00_clean.png'),clean)
                for n,img in variants.items():
                    cv2.imwrite(str(individual/f'{prefix}_{n}.png'),img)
                    rows.append({'sample':idx,'modality':modality,'degradation':profile,'method':n,'PSNR_dB':psnr(clean,img),'SSIM_global':ssim_global(clean,img),'MAE_8bit':np.abs(clean.astype(float)-img.astype(float)).mean(),'RMS_contrast':rms_contrast(img)})
                draw_sheet([('clean_reference',clean),*variants.items()],f'Real {modality} image {idx} | {profile} degradation',OUT/f'{prefix}_comparison.png')
                best_name, best_img = max(variants.items(), key=lambda pair: ssim_global(clean, pair[1]))
                draw_before_after(clean, bad, best_img,
                                  f'Real {modality} image {idx} | {profile} degradation',
                                  best_name.replace('_', ' '), featured/f'{prefix}_before_after.png')
    pd.DataFrame(rows).to_csv(OUT/'metrics_per_image.csv',index=False)
    summary=pd.DataFrame(rows).groupby(['modality','degradation','method'],as_index=False)[['PSNR_dB','SSIM_global','MAE_8bit','RMS_contrast']].mean()
    summary.to_csv(OUT/'metrics_summary.csv',index=False)
    print(f'Saved {len(list(OUT.glob("*_comparison.png")))} all-method sheets, {len(list(featured.glob("*.png")))} featured before/after panels, and {len(list(individual.glob("*.png")))} individual images to {OUT}')

if __name__=='__main__': main()
