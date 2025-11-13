# Dual-Color Space Dehazing: HSV-Guided Residual Removal and Color Fidelity Restoration(HSV-DehazeNet)

HSV-DehazeNet is a PyTorch-based image dehazing project that fuses RGB and HSV processing paths. The codebase includes model definitions, training and testing scripts, dataset utilities, loss functions, and evaluation metrics (PSNR / SSIM).

This README shows how to install dependencies, structure datasets, run training and inference, and points out important configuration notes.

## Repository layout

- `main.py` — training entry script
- `test.py` — simple inference script (edit paths inside)
- `data_utils.py` — dataset loader and data augmentation
- `option.py` — CLI argument parsing and interactive dataset selection
- `loss.py` — multi-term loss used for training
- `metrics.py` — PSNR / SSIM and some HSV/RGB helpers
- `models/HSVDehazeNet.py` — top-level network combining RGB and HSV branches
- `models/HSVNet.py` — HSV branch
- `models/RGBNet.py` — RGB branch

## Dependencies

- Python 3.8+
- PyTorch (install according to your CUDA version)
- torchvision
- kornia
- numpy
- pillow
- opencv-python
- matplotlib
- tqdm

Install example (PowerShell):

```powershell
python -m pip install torch torchvision kornia numpy pillow opencv-python matplotlib tqdm
```

Note: adjust the `torch` install line to match your CUDA toolkit if needed.

## Dataset format

The code expects a dataset root containing dataset folders. Each dataset folder should contain `train/` and `test/` subfolders, each with `hazy/` and `GT/` directories:

```
datasets_root/
  └─ YourDataset/
      ├─ train/
      │   ├─ hazy/
      │   └─ GT/
      └─ test/
          ├─ hazy/
          └─ GT/
```

`option.py` will list folders inside the `--path` directory and prompt you to choose one by number.

## Training

1. Place your dataset under the folder referenced by `--path` (or update the default in `option.py`).
2. Run training from the repository root. Example (PowerShell):

```powershell
python main.py --path "C:/path/to/datasets" --bs 2 --steps 30000 --crop
```

Important flags (from `option.py`):

- `--steps` — total training steps (default: 30000)
- `--device` — device (default: auto-detected CUDA if available)
- `--resume` — resume from existing checkpoint
- `--eval_step` — evaluation interval (default: 100)
- `--lr` — learning rate (default: 1e-4)
- `--bs` — batch size
- `--workers` — dataloader workers
- `--crop` / `--crop_size` — enable random cropping during training

During training the code will create `trained_models/`, `logs/` and `result/` directories as needed.

## Dataset
O-Haze：http://www.vision.ee.ethz.ch/ntire18/o-haze/ (Official from NTIRE18)
## Implementation notes

- Models:
  - `HSVNet` estimates HSV components and a fog-density-like map. It reconstructs an RGB output from HSV.
  - `RGBNet` is a multi-scale encoder-decoder with attention and frequency-aware blocks; it outputs multiple scales for multiscale supervision.
  - `HSVDehazeNet` fuses RGB and HSV branch outputs using an adaptive feature fusion (AFF) module.
- Loss: `loss.py` combines multi-scale L1 reconstruction losses, FFT-based frequency losses, and HSV-related losses.
- Metrics: `metrics.py` provides PSNR and SSIM metrics, and includes helper routines for color conversions.
