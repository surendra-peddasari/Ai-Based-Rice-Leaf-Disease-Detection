# AI-Based Rice Leaf Disease Detection

A deep learning and computer vision project for classifying rice leaf images into three disease categories:

- **Bacterial leaf blight**
- **Brown spot**
- **Leaf smut**

The repository contains the dataset, exploratory analysis, a Jupyter notebook for the main custom-CNN experiments, modular Python scripts, and generated evaluation/EDA artifacts.

> **Important implementation note:** The repository currently contains two related modeling tracks. The Jupyter notebook implements the documented **custom CNN baseline/tuned experiments**, while the scripts under `src/` implement a separate **MobileNetV2 transfer-learning pipeline**. The images in `Results/` are primarily artifacts from the custom-CNN notebook experiments. This README documents both tracks so the repository structure and actual code remain clear.

---

## 1. Project Overview

Rice is one of the world's most important food crops, and diseases affecting rice leaves can reduce plant health and yield. Early identification of visible disease symptoms can help farmers and agricultural practitioners take appropriate action.

This project explores an image-classification approach in which a rice-leaf photograph is processed by a convolutional neural network and assigned to one of three disease classes.

### Project objectives

1. Inspect and understand the available rice-leaf image dataset.
2. Examine the distribution and visual characteristics of the disease classes.
3. Preprocess and augment the images for deep-learning training.
4. Build a custom CNN suitable for the relatively small dataset.
5. Compare baseline and tuned custom-CNN behavior.
6. Evaluate predictions using a classification report and confusion matrix.
7. Provide a separate MobileNetV2 transfer-learning implementation for future/alternative experimentation.
8. Preserve generated plots and evaluation artifacts for analysis.

---

## 2. Disease Classes

| Class | Description |
|---|---|
| **Bacterial leaf blight** | A bacterial disease that produces characteristic blighting/necrotic symptoms on rice leaves. |
| **Brown spot** | A fungal disease associated with brown lesions/spots on rice leaves. |
| **Leaf smut** | A fungal disease characterized by dark smut-like lesions on affected leaves. |

The project should be treated as an **image-classification research/learning project**, not as a replacement for professional agricultural diagnosis.

---

## 3. Dataset

The dataset is stored directly inside the repository under:

```text
Ai-Based-Rice-Leaf-Disease-Detection/
└── Data/
    ├── Bacterial leaf blight/
    ├── Brown spot/
    └── Leaf smut/
```

### Dataset distribution

The repository contains **119 images** across the three classes:

| Disease class | Images |
|---|---:|
| Bacterial leaf blight | 40 |
| Brown spot | 40 |
| Leaf smut | 39 |
| **Total** | **119** |

The images are RGB photographs of rice leaves and are not all the same original resolution. The preprocessing pipeline resizes them before model training.

### Train/validation split

The notebook and `src/` pipelines use an approximately **80/20 train-validation split** through Keras `ImageDataGenerator`.

Because the dataset is small, validation results can vary substantially depending on the split and random initialization. Consequently, the reported validation performance should not be interpreted as a production-level accuracy estimate.

---

## 4. Exploratory Data Analysis

The notebook performs an initial inspection of:

- Number of images in each disease class.
- Potential class imbalance.
- Representative images from each class.
- Image dimensions.
- Sample augmented training images.

The generated EDA artifacts are stored in `Results/`:

```text
Results/
├── eda_class_distribution.png
├── eda_sample_images.png
└── sample_training_batch_grid.png
```

The dataset is relatively balanced by class (40, 40, and 39 images), so there is no severe class-count imbalance. The larger concern is the **small total number of training examples**.

---

## 5. Data Preprocessing and Augmentation

The notebook's custom-CNN pipeline uses:

- Pixel rescaling from `[0, 255]` to `[0, 1]`.
- Target image size: **128 × 128**.
- Batch size: **8**.
- Training/validation split: **80/20**.
- Rotation augmentation.
- Width and height shifts.
- Shearing.
- Zoom.
- Horizontal and vertical flips.
- `fill_mode='nearest'`.

The notebook also computes balanced class weights using:

```python
compute_class_weight(class_weight='balanced', ...)
```

These weights are supplied during custom-CNN training to reduce the effect of unequal class representation in the training subset.

### Why augmentation is used

The dataset is small, so a model can easily memorize training images rather than learn disease-related visual patterns. Augmentation creates different versions of the same training examples and exposes the model to changes in:

- Leaf orientation.
- Position within the image.
- Apparent scale.
- Small geometric distortions.

Augmentation does **not** create new independent biological samples, so it cannot fully compensate for a very small dataset.

---

## 6. Modeling Approaches

The repository contains two modeling approaches.

### 6.1 Custom CNN — notebook implementation

The notebook builds a compact CNN with three convolutional blocks:

```text
Input: 128 × 128 × 3
        │
        ├── Conv2D(16) + BatchNorm + MaxPooling + Dropout(0.2)
        │
        ├── Conv2D(32) + BatchNorm + MaxPooling + Dropout(0.3)
        │
        ├── Conv2D(64) + BatchNorm + MaxPooling + Dropout(0.4)
        │
        ├── Flatten
        ├── Dense(32, ReLU)
        ├── BatchNorm
        ├── Dropout(0.5)
        │
        └── Dense(3, Softmax)
```

The notebook describes two custom-CNN experiments:

1. **Baseline Custom CNN** — a larger model with approximately 4.28M parameters.
2. **Tuned Compact CNN** — a substantially smaller model designed to reduce over-parameterization on the small dataset.

The tuned model uses:

- Batch normalization.
- Dropout.
- Balanced class weights.
- `ReduceLROnPlateau`.
- `EarlyStopping`.
- Adam optimizer.
- Categorical cross-entropy loss.

The tuned model is saved by the notebook as:

```text
custom_rice_leaf_cnn_tuned.keras
```

The generated repository artifacts include:

```text
Results/
├── custom_cnn_confusion_matrix.png
├── custom_cnn_tuned_confusion_matrix.png
├── custom_cnn_finetuned_confusion_matrix.png
└── custom_cnn_training_curves.png
```

### 6.2 MobileNetV2 — modular `src/` implementation

The scripts under `src/` implement a separate transfer-learning pipeline using:

```text
MobileNetV2
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, ReLU)
    ↓
Dropout(0.5)
    ↓
Dense(3, Softmax)
```

Configuration used by the scripts:

| Parameter | Value |
|---|---|
| Backbone | MobileNetV2 |
| Pretrained weights | ImageNet |
| Base model | Frozen |
| Input size | 150 × 150 × 3 |
| Batch size | 8 |
| Epochs | 20 |
| Optimizer | Adam |
| Learning rate | 0.0001 |
| Loss | Categorical cross-entropy |
| Output classes | 3 |

This is a different experiment from the custom-CNN notebook and should not be confused with the custom-CNN result images.

---

## 7. Model Evaluation

The notebook evaluates the custom CNN using:

- Predicted class labels.
- Ground-truth validation labels.
- Classification report.
- Confusion matrix.

The confusion matrices make it possible to see which disease classes are being confused with one another.

### Existing recorded result artifacts

The repository includes a confusion matrix for the fine-tuned custom CNN:

```text
custom_cnn_finetuned_confusion_matrix.png
```

From the stored matrix, the validation predictions are:

| Actual \ Predicted | Bacterial leaf blight | Brown spot | Leaf smut |
|---|---:|---:|---:|
| Bacterial leaf blight | 8 | 0 | 0 |
| Brown spot | 0 | 3 | 4 |
| Leaf smut | 2 | 3 | 2 |

This corresponds to **13 correct predictions out of 22 validation images (≈59.1% accuracy)** for that stored experiment.

> This number describes the existing artifact only. It should not be treated as a stable benchmark because the dataset is very small and the repository does not provide a controlled multi-run statistical evaluation.

The baseline/tuned confusion matrices also demonstrate that some experiments strongly favored the **Brown spot** class, which is a useful indication of the difficulty of learning reliable class boundaries from such a small dataset.

---

## 8. Repository Structure

The actual project structure is:

```text
Ai-Based-Rice-Leaf-Disease-Detection/
│
├── Data/
│   ├── Bacterial leaf blight/
│   ├── Brown spot/
│   └── Leaf smut/
│
├── notebooks/
│   └── rice_leaf_detection.ipynb
│
├── Results/
│   ├── custom_cnn_confusion_matrix.png
│   ├── custom_cnn_finetuned_confusion_matrix.png
│   ├── custom_cnn_tuned_confusion_matrix.png
│   ├── custom_cnn_training_curves.png
│   ├── eda_class_distribution.png
│   ├── eda_sample_images.png
│   └── sample_training_batch_grid.png
│
├── src/
│   ├── evaluation_metrics.py
│   ├── inference.py
│   ├── inspecting.py
│   ├── lib.py
│   ├── plot.py
│   ├── preprocess.py
│   ├── report.py
│   └── train.py
│
├── requirements.txt
└── README.md
```

### Purpose of important files

| File | Purpose |
|---|---|
| `notebooks/rice_leaf_detection.ipynb` | Main exploratory analysis, custom CNN experiments, tuning, evaluation, and discussion. |
| `src/lib.py` | Dataset download/extraction helper and TensorFlow setup. |
| `src/inspecting.py` | Counts images per class and checks whether images can be opened by PIL. |
| `src/preprocess.py` | Creates training and validation generators with augmentation. |
| `src/train.py` | Builds and trains the MobileNetV2 transfer-learning model and saves `rice_leaf_model.h5`. |
| `src/evaluation_metrics.py` | Evaluates a saved MobileNetV2 model using validation data. |
| `src/inference.py` | Currently performs a MobileNetV2 training/evaluation workflow and generates plots; despite its filename, it is not a single-image inference script. |
| `src/plot.py` | Another MobileNetV2 training/evaluation and plotting workflow. |
| `src/report.py` | Generates a formatted Word technical report and associated plots. |
| `requirements.txt` | Python dependencies. |

---

## 9. Installation

### 9.1 Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Ai-Based-Rice-Leaf-Disease-Detection
```

If you already downloaded the repository as a ZIP file, extract it and enter the project directory.

### 9.2 Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 9.3 Install dependencies

```bash
pip install -r requirements.txt
```

TensorFlow installation can be platform-dependent. If the standard installation fails, use the TensorFlow installation instructions appropriate for your operating system and Python version.

---

## 10. Running the Project

### Option A — Run the notebook

The notebook is the best place to reproduce the documented custom-CNN analysis.

```bash
jupyter notebook notebooks/rice_leaf_detection.ipynb
```

Then run the notebook cells in order:

```text
EDA
  ↓
Preprocessing & Augmentation
  ↓
Custom CNN
  ↓
Training
  ↓
Evaluation
  ↓
Confusion Matrix
  ↓
Conclusion
```

### Option B — Inspect the dataset

From the project root:

```bash
python src/inspecting.py
```

This checks the image counts and attempts to verify that each image can be read.

### Option C — Run preprocessing

```bash
python src/preprocess.py
```

This creates the training and validation image generators and prints the discovered class mapping.

### Option D — Train the MobileNetV2 model

```bash
python src/train.py
```

The script trains the transfer-learning model for 20 epochs and saves:

```text
rice_leaf_model.h5
```

> The scripts expect the dataset path `rice_leaf_dataset/Data`. If the repository's `Data/` directory is used directly instead, update `DATA_DIR` in the relevant script or place the dataset at the expected path.

---

## 11. Dataset Download Helper

`src/lib.py` contains the original dataset download/extraction logic.

It downloads the dataset archive and extracts it into:

```text
rice_leaf_dataset/
```

The script references the original project dataset URL.

If you already have the dataset in the repository, downloading it again is unnecessary.

---

## 12. Evaluation Interpretation

A high training accuracy alone does not prove that the model has learned disease characteristics.

For this project, special attention should be given to:

- Validation accuracy.
- Validation loss.
- Confusion matrix.
- Per-class precision.
- Per-class recall.
- Per-class F1-score.
- Whether the model predicts one class disproportionately often.

Because there are only 119 images, a difference of only a few validation predictions can change the reported accuracy considerably.

For example, a model that predicts nearly every image as one class may still obtain a non-trivial accuracy if that class has several validation examples, while performing poorly on the remaining diseases. The confusion matrix therefore provides important information that accuracy alone cannot show.

---

## 13. Limitations

### Small dataset

The biggest limitation is the very small number of images. Deep-learning models generally benefit from much larger and more diverse datasets.

### Limited validation set

An approximately 20% validation split produces only around 22–24 validation images. This makes the evaluation statistically unstable.

### Potential overfitting

The custom CNN experiments show that model capacity can become too large relative to the available training data.

### Dataset diversity

The repository does not establish that the images cover the full range of:

- Rice varieties.
- Lighting conditions.
- Camera devices.
- Leaf orientations.
- Disease severity levels.
- Field backgrounds.
- Geographic growing conditions.

### No production deployment layer

The repository currently focuses on model development and evaluation. It does not contain a complete web/mobile application, REST API, authentication system, or production monitoring pipeline.

### `src/inference.py` naming

Despite its name, `src/inference.py` currently retrains a MobileNetV2 model and generates evaluation plots. It does not provide a simple function such as:

```python
predict("path/to/leaf.jpg")
```

A production-ready inference module would be a useful future improvement.

---

## 14. Recommended Future Improvements

1. **Increase dataset size** with more independently collected images.
2. **Use stratified train/validation/test splits** so the final test set is never used during model selection.
3. **Use repeated experiments or cross-validation** to obtain more reliable performance estimates.
4. **Standardize the modeling pipeline** so the notebook and `src/` scripts use the same preprocessing, model, and evaluation procedure.
5. **Add a true inference script** for predicting one uploaded image.
6. **Save class labels** explicitly with the trained model or in a configuration file.
7. **Use transfer learning consistently** for small datasets, with careful fine-tuning of selected backbone layers.
8. **Add precision, recall, F1-score, and ROC/PR analysis** where appropriate.
9. **Track experiments** with fixed random seeds and saved training histories.
10. **Add a held-out test set** that is never used for tuning.
11. **Add explainability tools**, such as Grad-CAM, to visualize which leaf regions influence predictions.
12. **Package the model into an API or web application** for practical image-based prediction.
13. **Add automated tests and CI** for data loading, preprocessing, and inference.
14. **Avoid storing large raw datasets in Git** if repository size becomes a concern; use Git LFS or an external dataset store.

---

## 15. Reproducibility Notes

For reproducible experiments, it is recommended to:

- Record the Python version.
- Record the TensorFlow/Keras version.
- Fix random seeds.
- Keep a fixed train/validation/test split.
- Save the exact trained model.
- Save the training history.
- Record class-to-index mappings.
- Record the preprocessing and augmentation configuration.

The current repository does not completely standardize these reproducibility controls, so results may vary between runs.

---

## 16. Technologies Used

- **Python**
- **TensorFlow / Keras**
- **MobileNetV2**
- **Convolutional Neural Networks**
- **NumPy**
- **Pandas**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Pillow**
- **Jupyter Notebook**

---

## 17. Project Workflow

```text
                    Rice Leaf Images
                           │
                           ▼
                 Dataset Inspection
                           │
                           ▼
                    Exploratory EDA
                           │
                           ▼
             Resize + Normalize Images
                           │
                           ▼
             Data Augmentation (Train)
                           │
                           ▼
                ┌──────────┴──────────┐
                │                     │
                ▼                     ▼
         Custom CNN Track       MobileNetV2 Track
         (Notebook)              (src/)
                │                     │
                ▼                     ▼
          Train + Tune             Train
                │                     │
                └──────────┬──────────┘
                           ▼
                       Evaluation
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
      Classification Report       Confusion Matrix
             │                           │
             └─────────────┬─────────────┘
                           ▼
                    Model Analysis
                           │
                           ▼
                  Future Improvements
```

---

## 18. Conclusion

This project demonstrates an end-to-end workflow for rice leaf disease image classification, from dataset inspection and augmentation through CNN training and evaluation.

The experiments also demonstrate an important machine-learning lesson: **model complexity must match dataset size**. The repository's custom-CNN experiments show how a high-capacity model can struggle with a small dataset, while the compact architecture introduces stronger regularization and reduced parameter count.

The separate MobileNetV2 implementation provides a transfer-learning direction that is particularly relevant when the number of labeled agricultural images is limited.

For a production-quality disease detection system, the next major priorities should be a substantially larger and more diverse dataset, a strictly separated test set, reproducible experiments, a standardized model pipeline, and a dedicated single-image inference/deployment layer.
