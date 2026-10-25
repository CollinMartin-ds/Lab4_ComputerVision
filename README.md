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

On Google Colab, set `COLAB_ARCHIVE_PATH` in the notebook to the archive's location in Google Drive.

## Requirements

Python 3 with:

```bash
pip install torch torchvision numpy pandas scikit-image scikit-learn matplotlib pillow
```

A GPU is recommended for training on the full split. Use `maxSize` in the dataset class to train on a smaller subset while you test code.

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
- [ ] Set `MAX_TRAIN_SAMPLES = 0`, run the baseline notebook and submit it. This checks the ZIP format and gives a reference score.

### 2. Notebook infrastructure (days 1–2)
- [ ] Cache all decoded images in RAM (~1.8 GB) instead of re-reading JPEGs every epoch.
- [ ] Raise `NUM_WORKERS` to 4–8, set `pin_memory=True` and use mixed precision (AMP).
- [ ] `train_model`:
  - [ ] Log loss and macro F1 per epoch (needed for the report's curves).
  - [ ] Save the best checkpoint to disk.
  - [ ] Support AdamW and cosine/OneCycle schedules.
- [ ] **Test-time augmentation (TTA):** average predictions over the 8 flip/90° rotation views of each image.
- [ ] **Macro-F1 calibration:** find per-class probability weights on val that maximise macro F1, apply them to test and renormalise.
- [ ] Keep an experiment log: category, change, epochs, val macro F1.

### 3. FINE-TUNING model (days 2–7)
- [ ] Train AlexNet on the full data as the reference row.
- [ ] Try `timm` backbones: `convnext_tiny`/`convnext_small`, `efficientnetv2_s`, `swin_tiny`. A remote-sensing pretrained ViT is worth trying too.
- [ ] Train and predict at the **native 320 px**. Don't center-crop to 224.
- [ ] Recipe:
  - AdamW, lr ~1e-4 for the backbone and ~1e-3 for the head;
  - cosine schedule with warmup, 20–30 epochs;
  - label smoothing 0.1.
- [ ] Imbalance: compare class weights ∝ 1/√frequency, a weighted sampler, and plain cross-entropy + calibration.
- [ ] Try mixup/CutMix.
- [ ] Optional ensemble of 2–3 models. Ask the professor first whether it's allowed.

### 4. CUSTOM model (days 4–10)
- [ ] Train the reference net on the full data as the baseline.
- [ ] Add BatchNorm and use deeper stacks of 3×3 convolutions (32→64→128→256). Use **global average pooling** + dropout instead of the huge `fc1`.
- [ ] Build a small ResNet by hand from primitive layers only.
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
