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
