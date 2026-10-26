# Rice Leaf Disease Detection 

 

---

## Project Overview
This repository contains an end-to-end deep learning pipeline for detecting and classifying rice leaf diseases to assist in automated agricultural disease diagnosis.

## Repository Structure
```text
PRCP-1001-Rice-Leaf-Disease-Detection/
│
├── data/Bacterial leaf blight                      # Dataset directory (raw & processed split into train/val/test)
├── notebooks/                 # Jupyter Notebooks (EDA, Model Training, Evaluation)
│   └── PRCP_1001_Rice_Leaf_Disease.ipynb
├── src/                       # Source python scripts for modular pipelines
│   ├── __init__.py
│   ├── data_preprocessing.py  # Image augmentation and loading generators
│   ├── model.py               # CNN / Transfer learning architecture definition
│   ├── train.py               # Training loop with callbacks and history logging
│   └── evaluate.py            # Evaluation metrics, confusion matrix, and classification report
├── outputs/                   # Saved models, weights, and generated plots
├── requirements.txt           # Project dependencies
└── README.md                  # Project documentation
```

## Setup & Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/PRCP-1001-Rice-Leaf-Disease-Detection.git
   cd PRCP-1001-Rice-Leaf-Disease-Detection
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the notebook or modular Python scripts in `src/`.
