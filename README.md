# Lab 4: Predicting Fire Risk from Satellite Images

Lab session 4 for *Audio Processing, Video Processing and Computer Vision* (BSc in Data Science and Engineering, Universidad Carlos III de Madrid, 2026/2027). This project includes a report and a CodaBench competition.

The goal is to classify satellite images into seven fire-risk classes using convolutional neural networks in PyTorch:

| Index | Class        |
|-------|--------------|
| 0     | Water        |
| 1     | Very Low     |
| 2     | Low          |
| 3     | Moderate     |
| 4     | High         |
| 5     | Very High    |
| 6     | Non-burnable |

## Contents

| File | Description |
|------|-------------|
| `Lab_4_AVCV_2026_2027_Students.ipynb` | Main notebook: dataset, augmentation, models, training and assessment |
| `Lab4_AVCV_2026_2027_Report_Guidelines.pdf` | Requirements for the written report |
| `Lab4_AVCV_2026_2027_CodaBench_Pair_Submission_Guide.pdf` | How to submit predictions to CodaBench as a pair |

## Notebook outline

- **Part 0:** The problem and the dataset
- **Part 1:** A custom PyTorch `Dataset`, preprocessing and data augmentation (flips, 90° rotations, crops, colour jitter)
- **Part 2:** Building `CustomNet`, a CNN written from scratch
- **Part 3:** Training: loss, optimiser, learning-rate schedule and training loop
- **Part 4:** Fine-tuning a pretrained `torchvision` network
- **Part 5:** Assessment and generating competition predictions

## Dataset

The dataset (`db.zip`, about 920 MB, 5,861 JPEG images) is **not included in this repository** because it exceeds GitHub's file size limit. Get it from the course materials and put it in the repository root. The notebook extracts it to:

```
db/
├── images/
├── train_reduced.csv
├── val_reduced.csv
└── test_reduced.csv
```

On Google Colab, the notebook mounts Google Drive and reads `db.zip` from `DRIVE_DIR` (currently `/content/drive/MyDrive/lab_4_CV`). Change `DRIVE_DIR` in the Colab setup cell if your copy is somewhere else. Outputs are copied back to `DRIVE_DIR/outputs/` at the end.

## Requirements

Python 3 with:

```bash
pip install torch torchvision numpy pandas scikit-image scikit-learn matplotlib pillow
```

A GPU is recommended for training on the full split. Use `maxSize` in the dataset class to train on a smaller subset while you test code.

## Original baseline (professor's handout)

This is the starting point every later run is compared against. It was checked cell by cell against the unmodified handout notebook (`Lab_4_Original.ipynb`, not in the repo). The repo's first commit (`41aad7f`) is **not** the untouched handout. The table below describes the handout itself and notes where `41aad7f` differs.

| Setting | Handout value |
|---|---|
| Training data | `MAX_TRAIN_SAMPLES = 500` (a reproducible random subset of the 3,516 training images) |
| Epochs | `CUSTOM_EPOCHS = 5`, `FT_EPOCHS = 5` |
| Training transform | `RandomCrop(224)` → `ToTensor` → ImageNet `Normalize`. **Crop-only augmentation**: no flips, rotations or jitter. |
| Val/test transform | `CenterCrop(224)` → `ToTensor` → `Normalize` |
| Batches | 64 for training, 128 for val/test, `NUM_WORKERS = 0`, `SEED = 42` |
| Loss | plain `nn.CrossEntropyLoss()`: no class weights, no label smoothing |
| Training loop | keeps the best-val-macro-F1 checkpoint, runs every epoch, no TTA, saves no weights |

**Original CUSTOM.** In the handout, `CustomNet.__init__` is an exercise that raises `NotImplementedError`. Its `forward()` and the comments fix the reference architecture, which `41aad7f` implements as follows:
- `Conv2d(3, 6, 5)` → ReLU → `MaxPool2d(2)` (224 → 110)
- `Conv2d(6, 16, 5)` → ReLU → `MaxPool2d(2)` (→ 53)
- `Conv2d(16, 32, 5)` → ReLU → `MaxPool2d(2)` (→ 24)
- flatten (32·24·24 = 18,432) → `Linear(18432, 120)` → ReLU → `Linear(120, 7)`
- No BatchNorm, dropout or global pooling. Most of the parameters are in `fc1`.
- SGD, lr `1e-2`, momentum 0.9, `StepLR(step_size=7, gamma=0.1)`. The step never fires in 5 epochs.

**Original FINE-TUNING.** torchvision AlexNet (`IMAGENET1K_V1`), with `classifier[6]` replaced by `Linear(4096, 7)`. All weights are fine-tuned with SGD, lr `1e-3`, momentum 0.9, `StepLR(7, 0.1)`.

**What `41aad7f` already changed relative to the handout:**
- It filled in the `CustomNet` layers listed above.
- It added the `ApplyToImage` and `RandomRotate90` transforms, plus the gallery of 32 augmentation combinations and 24 operation orders.
- It switched training augmentation to `RandomCrop(224)` + H/V flips + rot90 + `ColorJitter(0.1, 0.1, 0.05)`.
- It replaced the handout's demo `composed` pipeline (jitter with hue 0.01, flips) and filled the empty transform exercise cells.
- It rewrote the intro note about the unfinished exercise and added a small `ToTensor` demo.
- Everything from the metric/loss section onward (loss, optimisers, `train_model`, AlexNet, test, submission) is identical to the handout.

## Experiment log

| Run | Date | Commit | Category | Model and recipe | Epochs | Best val macro F1 | Test macro F1 (CodaBench) |
|-----|------|--------|----------|------------------|--------|-------------------|---------------------------|
| 1 | 2026-11-04/06 | `e3fc418` + uncommitted outputs | CUSTOM | 5-block BN CNN + GAP, AdamW, cosine | 25 | 0.4385 (epoch 24) | ≈0.44 |
| 1 | 2026-11-04/06 | `e3fc418` + uncommitted outputs | FINE-TUNING | ConvNeXt-Tiny (ImageNet), SGD, StepLR | 5 | 0.4613 (epoch 3) | ≈0.44 |
| 1b | 2026-11-06 | same as run 1 | CUSTOM | Run-1 CustomNet, same recipe | 60 | not recorded | ≈0.43 |
| 2 | 2026-11-06 | repo notebook (uncommitted) with `CUSTOM_EPOCHS = 50` | CUSTOM | Residual CustomNet, 288 px crops / 320 px eval, AdamW + OneCycle (3e-3), AMP, TTA + calibration | 50 | not recorded | **0.49** |
| 2 | 2026-11-06 | same | FINE-TUNING | ConvNeXt-Tiny (stoch. depth 0.2), 288/320 px, AdamW 1e-4/1e-3 + OneCycle, AMP, TTA + calibration | 20 | not recorded | **0.50** |
| side | 2026-11-06 | Colab copy only | both | ResNet-50 / run-1 CustomNet at 224 px resize (see below) | — | not run | not submitted |

**Run 1b conclusion:** training the run-1 custom net for 60 instead of 25 epochs did not help (test 0.43 vs 0.44). It had plateaued, so the bottleneck is the architecture, input resolution and recipe, not the epoch budget.

### Run 1: full configuration

**Environment:** Google Colab, `cuda:0`, `db.zip` read from Google Drive. `SEED = 42`, deterministic cuDNN, `NUM_WORKERS = 0`, `MAX_TRAIN_SAMPLES = 0` (full split: 3,516 train / 1,172 val / 1,173 test).

**Data pipeline (shared by both models):**
- Training: `RandomCrop(224)` from the 320 px image → `ToTensor` → random horizontal flip, vertical flip and 90° rotation → `ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05)` → ImageNet mean/std normalisation.
- Validation and test: `CenterCrop(224)` → `ToTensor` → normalisation. No TTA.
- Batch size 64 for training, 128 for validation and test. Images are re-read from JPEG on every access.
- The notebook also renders an augmentation gallery (all 32 on/off combinations of C/H/V/R/J for each class, and all 24 operation orders). This is a visual check only, not a validation experiment.

**Loss (shared):** `CrossEntropyLoss` with class weights ∝ 1/√(train frequency), normalised to mean 1 (`[1.899, 0.534, 0.761, 0.848, 0.992, 1.379, 0.588]`), and label smoothing 0.1.

**Training loop (shared):** runs every epoch (no early stopping), restores the checkpoint with the best validation macro F1 and plots its row-normalised confusion matrix. Nothing is saved to disk except the output CSVs and ZIP.

**CUSTOM model:** `CustomNet`, built from scratch:
- 5 blocks of [Conv3×3 → BN → ReLU] ×2 → MaxPool2d(2), with channels 32 → 64 → 128 → 256 → 256 (224 → 7 px).
- `AdaptiveAvgPool2d(1)` → `Dropout(0.3)` → `Linear(256, 7)`.
- AdamW, lr 1e-3, weight decay 0.05, `CosineAnnealingLR(T_max=25)`, 25 epochs. Took 25 min 53 s.
- Val F1 was noisy (0.19 → 0.41 → 0.33) during the first half, then rose steadily to 0.4385 at the **final** epoch. Train F1 (0.41) stayed below val F1, so the model was **underfitting**: it stopped while still improving.

**FINE-TUNING model:** torchvision `convnext_tiny` with `IMAGENET1K_V1` weights (27.8M parameters). The final classifier layer was replaced with `Linear(768, 7)` and the whole network was fine-tuned.
- SGD, lr 1e-3, momentum 0.9, `StepLR(step_size=7, gamma=0.1)`, 5 epochs. Took 8 min 43 s.
- With only 5 epochs, `StepLR` never fired, so the learning rate stayed constant. That optimiser and schedule were inherited from the AlexNet baseline and are not tuned for ConvNeXt.
- Val F1: 0.40, 0.46, 0.43, **0.4613**, 0.45. Train F1 reached only 0.47, so this model was also **underfitting**.

**Submission:** softmax probabilities from the restored best checkpoints, saved as `output_custom.csv` and `output_ft.csv` (1173 × 7), zipped into `codabench_submission.zip` and copied to Drive. Macro F1 on CodaBench was about 0.44.

**Known issues with run 1:**
- No model weights were saved (`torch.save`), so the submission can't be regenerated bit-for-bit.
- Some markdown cells are out of date: cell 56 says "60 epochs" (the run used 25), and cells 58 and 64 still describe AlexNet.
- `CenterCrop(224)` discards about half of each 320 px image at val and test time.

### Run 2: the repo notebook, run on Colab (best so far: 0.49 / 0.50)

Run 2 is `Lab_4_AVCV_2026_2027_Students.ipynb` from this repo with the changes listed under "Run 2 changes" below. In Colab, `CUSTOM_EPOCHS` was set to 50 (the notebook now says 50 too) and `FT_EPOCHS` stayed at 20.

**Scores (CodaBench test macro F1):**
- CUSTOM **0.49**, up from 0.44 (run 1) and 0.43 (run 1b).
- FINE-TUNING **0.50**, up from 0.44 (run 1).

**Evidence:** the outputs are in Drive at `MyDrive/lab_4_CV/outputs/Lab_4_2026_2027/`.
- Each checkpoint matches its architecture: `custom_best.pt` (11.2 MB, the 2.8M-parameter residual CustomNet) and `ft_best.pt` (111 MB, ConvNeXt-Tiny).
- Both were last written at 12:42 / 12:51 UTC on Nov 6.
- Also there: `class_weights_custom.npy` / `class_weights_ft.npy` (calibration), `output_custom.csv`, `output_ft.csv` and `codabench_submission.zip`, all written at 12:57 UTC. Downloads has the same zip as `codabench_submission (3).zip`.

**Reproducing it:** in the repo notebook, set `LOAD_CHECKPOINTS = True` and run all cells on Colab. The notebook loads the two checkpoints from that Drive folder, skips training, and redoes TTA + calibration, so it regenerates the same CSVs.

**Not recorded:** the per-epoch curves and the plain / TTA / calibrated val F1 values. The executed notebook wasn't saved (the copy in Drive is the unrun upload). The `LOAD_CHECKPOINTS = True` run above prints the val numbers again, so it fills this gap.

### Side experiment: ResNet-50 Colab variant (drafted, not run)

This variant was edited in Colab, separately from the repo notebook. The copy is `~/Downloads/Lab_4_AVCV_2026_2027_Students (1).ipynb` (downloaded 2026-11-06 23:18). The changes, compared with commit `e3fc418`:

**Setup**
- Added a `SMOKE` switch: `True` gives 128 images and 1 epoch, `False` the full run.
- `INPUT_SIZE = 224`, `BATCH_SIZE = 32` (down from 64), `USE_CLASS_WEIGHTS = True`.
- `CUSTOM_EPOCHS = 60`, `FT_EPOCHS = 40`, `FT_LR = 1e-3`, new `FT_WEIGHT_DECAY = 1e-3`.
- `NUM_WORKERS = 2` and `pin_memory=True` on CUDA (0 / False elsewhere).

**Data**
- `FireRiskDataset` decodes every image once and caches it in RAM as uint8.
- Training no longer crops: the whole 320 px tile is **resized** to 224 (`Rescale((224, 224))`), then the same flips, rot90 and colour jitter as run 1. This follows Shen et al. and the Stanford FireRisk study, which found flips help ResNet and random crops don't.
- Val and test are also resized to 224 instead of `CenterCrop(224)`, so the model sees the whole scene.
- Val and test batches are 32 (down from 128). The training-batch preview was removed.

**Loss**
- The two models get separate losses.
  - CUSTOM keeps the run-1 loss: weights ∝ 1/√frequency, now computed from the full train CSV, plus label smoothing 0.1.
  - FINE-TUNING uses **plain cross-entropy** with no class weights and no smoothing, as in the published FireRisk fine-tunes.

**Training loop**
- AMP (fp16 autocast + `GradScaler`) during training on CUDA.
- Per-epoch history of loss and F1, with loss and F1 curves plotted at the end.
- The best checkpoint is saved as a dict (`model_state_dict`, `epoch`, `best_f1`, `history`) to `custom_best.pt` / `ft_best.pt`.
- Removed the input checks: criterion type, empty loaders, logits shape, non-finite loss.

**CUSTOM model**
- Architecture unchanged from run 1 (5 plain BN conv blocks + GAP).
- Same AdamW 1e-3 / wd 0.05 / cosine recipe, now for 60 epochs at batch 32 on resized tiles.

**FINE-TUNING model**
- Switched from ConvNeXt-Tiny to **ResNet-50** (ImageNet-1K V1), the supervised benchmark in Shen et al. (2023), with a new `fc` layer of 7 outputs.
- SGD (momentum 0.9) at lr 1e-3, weight decay 1e-3, on all weights.
- `CosineAnnealingLR(T_max=40)` replaces StepLR. 40 epochs, keeping the best val checkpoint.

**Prediction**
- `test_model` averages softmax over the 8 flip/rot90 views (TTA) for any loader.
- New `calibrate_probabilities`: coordinate ascent over per-class multipliers `[0.5, 0.75, 1, 1.25, 1.5, 2, 3]`, 2 passes, fitted on val TTA probabilities and applied to test.
- Prints single-view, TTA and calibrated val macro F1 for both models.
- New last cell copies the CSVs, zip and `.pt` files to `MyDrive/Lab4_AVCV/outputs`. Note this is a different folder from the `MyDrive/lab_4_CV` the data comes from.
- The "Part 5: Assessment" markdown was replaced with an experiment-log table whose values are still TBD.

**Status:** not trained or scored.
- In the downloaded copy, the setup, training and evaluation cells have no outputs or execution count.
- Its final cell would have created `MyDrive/Lab4_AVCV/outputs`, and that folder doesn't exist in Drive.
- `codabench_submission (3).zip` in Downloads sits next to this notebook, but it is the run-2 submission from the repo notebook (same files as in Drive), not output of this variant.
- The ideas in it are still worth testing: plain cross-entropy for fine-tuning, ResNet-50, and resizing the whole tile to 224.
### Run 2 changes (implemented in the repo notebook)

Each item below is in `Lab_4_AVCV_2026_2027_Students.ipynb` in this repo; the exact settings are in the setup cell. Run 2 was trained with `CUSTOM_EPOCHS = 50` and `FT_EPOCHS = 20`.

FINE-TUNING (highest expected gain):
- AdamW with layer-wise learning rates (backbone 1e-4, head 1e-3), weight decay 0.05, warmup + cosine/OneCycle, 20–25 epochs.
- Train on 288 px random crops and predict on the full 320 px image (no center crop).
- Stochastic depth raised to 0.2 (torchvision's default for Tiny, already used in run 1, is 0.1).
- Not done yet (ideas for run 3): EMA of the weights, a stronger backbone (`convnext_small`, `efficientnet_v2_s`, `swin_t`).

CUSTOM:
- Faster data loading: images are decoded once and cached in RAM as uint8; `NUM_WORKERS=2` and `pin_memory=True` on CUDA; AMP (fp16). 50 epochs.
- OneCycleLR with a peak lr of about 3e-3.
- Hand-built residual network (`ResidualBlock`): strided stem, then 4 stages × 2 blocks with 32/64/128/256 channels, GAP, dropout 0.3 and a linear head. About 2.8M parameters, primitive layers only.
- Train on 288 px random crops and predict on the full 320 px image.

Both:
- 8-view flip/rot90 TTA.
- Per-class probability weights tuned on val to maximise macro F1.
- The best weights are saved to `custom_best.pt` / `ft_best.pt` (on Colab, straight into the Drive output folder). Set `LOAD_CHECKPOINTS = True` to regenerate the submission from them without retraining.
- Per-epoch loss, F1 and lr are returned in `history`, and `plot_history` draws the learning curves.
- The prediction cell prints val macro F1 as plain / TTA / TTA + calibration for each model, so every step's effect can be reported.

## Roadmap to win the challenge

**What matters:**
- **Grade:** 45% report, 45% leaderboard ranking, 10% code.
- **Metric:** macro F1, which weights all 7 classes equally. Train is very imbalanced (Water 86, Very High 163 vs Very Low 1,088), so **the rare classes decide the score**.
- **Limits:** 2 submissions per day. Deadline **Nov 9, 23:59**.
- **Reproducibility:** if the notebook can't reproduce what was submitted, the team is disqualified from the ranking.

**Validation macro-F1 targets:** CUSTOM ≈ 0.40–0.50, FINE-TUNING ≈ 0.50–0.60.

### 1. Setup (day 1)
- [ ] Create the CodaBench organization `ID1_ID2`, invite your partner, and always pick **Submit as → team**.
- [ ] Set up the GPU environment: torch + CUDA, `pip install timm`, copy `db.zip` across.
- [x] Set `MAX_TRAIN_SAMPLES = 0`, run the baseline notebook and submit it. This checks the ZIP format and gives a reference score. (Run 1, test macro F1 ≈ 0.44.)

### 2. Notebook infrastructure (days 1–2)
- [x] Cache all decoded images in RAM (~1.8 GB) instead of re-reading JPEGs every epoch.
- [x] Raise `NUM_WORKERS` (2 on Colab), set `pin_memory=True` and use mixed precision (AMP).
- [x] `train_model`:
  - [x] Log loss and macro F1 per epoch (needed for the report's curves).
  - [x] Save the best checkpoint to disk.
  - [x] Support AdamW and cosine/OneCycle schedules.
- [x] **Test-time augmentation (TTA):** average predictions over the 8 flip/90° rotation views of each image.
- [x] **Macro-F1 calibration:** find per-class probability weights on val that maximise macro F1, apply them to test and renormalise.
- [x] Keep an experiment log: category, change, epochs, val macro F1.

### 3. FINE-TUNING model (days 2–7)
- [ ] Train AlexNet on the full data as the reference row.
- [ ] Try `timm` backbones: `convnext_tiny`/`convnext_small`, `efficientnetv2_s`, `swin_tiny`. A remote-sensing pretrained ViT is worth trying too.
- [x] Train and predict at the **native 320 px**. Don't center-crop to 224.
- [ ] Recipe:
  - AdamW, lr ~1e-4 for the backbone and ~1e-3 for the head;
  - cosine schedule with warmup, 20–30 epochs;
  - label smoothing 0.1.
- [ ] Imbalance: compare class weights ∝ 1/√frequency, a weighted sampler, and plain cross-entropy + calibration.
- [ ] Try mixup/CutMix.
- [ ] Optional ensemble of 2–3 models. Ask the professor first whether it's allowed.

### 4. CUSTOM model (days 4–10)
- [ ] Train the reference net on the full data as the baseline.
- [x] Add BatchNorm and use deeper stacks of 3×3 convolutions (32→64→128→256). Use **global average pooling** + dropout instead of the huge `fc1`. (Run 1: 5 blocks, 32→64→128→256→256.)
- [x] Build a small ResNet by hand from primitive layers only.
- [ ] Train for **60–100 epochs** with AdamW and OneCycle/cosine.
- [ ] Use the same augmentation, imbalance handling, TTA and calibration as for fine-tuning.

### 5. Submitting (ongoing)
- [ ] Only submit when val F1 clearly improves. Don't chase the public leaderboard; it's a small undisclosed subset.
- [ ] Record which commit or checkpoint produced each submission.

### 6. Lock it down (days 11–12)
- [ ] From a fresh kernel, run the final notebook end to end and confirm it regenerates the submitted CSVs.

### 7. Report (days 12–14): 3 pages + 1 page of figures and tables
- [ ] Follow `Lab4_AVCV_2026_2027_Report_Guidelines.pdf` section by section.
- [ ] Include:
  - [ ] experiment table;
  - [ ] loss and F1 curves;
  - [ ] row-normalised confusion matrices;
  - [ ] misclassified validation images;
  - [ ] public scores for both categories;
  - [ ] who did what.
- [ ] Cite the FireRisk paper (Shen et al., 2023, arXiv:2303.07035) and the backbones you used.

### 8. Hand in (by Nov 8, leaving a buffer day)
- [ ] Submit the final ZIP on CodaBench as the team.
- [ ] Upload the report + notebook ZIP to Aula Global.
