# HSV-DehazeNet

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
python main.py --path "C:/path/to/datasets" --net HSVDehazeNet --bs 2 --steps 30000
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

## Inference / Testing

`test.py` demonstrates loading a saved `HSVNet` model and running predictions. Edit `img_dir` and `model_dir` at the top of `test.py`, then run:

```powershell
python test.py
```

Predicted images are written to a folder named like `pred_HSVNet_<dataset>/` (see `test.py`).

## Implementation notes

- Models:
  - `HSVNet` estimates HSV components and a fog-density-like map. It reconstructs an RGB output from HSV.
  - `RGBNet` is a multi-scale encoder-decoder with attention and frequency-aware blocks; it outputs multiple scales for multiscale supervision.
  - `HSVDehazeNet` fuses RGB and HSV branch outputs using an adaptive feature fusion (AFF) module.
- Loss: `loss.py` combines multi-scale L1 reconstruction losses, FFT-based frequency losses, and HSV-related losses.
- Metrics: `metrics.py` provides PSNR and SSIM metrics, and includes helper routines for color conversions.

## Known caveats & recommendations

- Some dataset-dependent mean/std values are used in `metrics.py` and `test.py` to denormalize/normalize images. Ensure `opt.trainset` is set consistently (or add your dataset's mean/std where needed).
- `option.py` prompts interactively for dataset selection; for scripted runs you can set `opt.dataset_name` programmatically before use.
- A few comments or strings in code may still be in Chinese. If you want a full sweep to translate all text to English, I can perform that.

## Suggested next improvements

1. Add a non-interactive CLI argument to choose dataset name to support headless execution.
2. Centralize dataset mean/std configuration (e.g., in `option.py`) to avoid duplication.
3. Provide a small example dataset and a reproducible training example to verify the pipeline.

## License

No license file is included. Add a `LICENSE` if you plan to publish or share the code publicly.

---

If you want, I can now:

- translate any remaining Chinese comments and strings across the repo, or
- add a non-interactive dataset selection flag, or
- create a small script that runs a quick smoke test (if you install PyTorch locally).

