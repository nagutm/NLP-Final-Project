# NLP Final Project

This repository contains code, datasets, and utilities for training, evaluating, and running an extractive and refinement-based NLP question-answering pipeline. The project includes end-to-end workflows for data preparation, model training, prediction, and evaluation.

---

## 📁 Project Structure
```
NLP-Final-Project/
│
├── README.md                 # Project overview and instructions
├── requirements.txt          # Python dependencies
├── datasets/                 # Processed train/val/test JSONL datasets
├── notebooks/
│   └── data_exploration.ipynb
├── src/
│   ├── extractive_predictor.py
│   ├── run_predictor_demo_extractive.py
│   ├── run_predictor_demo_refinement.py
│   ├── train_extractive.py
│   ├── train_refinement.py
│   └── utils.py

```

---

## 🚀 Setup Instructions
### 1. **Create a virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate    # Windows
```

### 2. **Install dependencies**
```bash
pip install -r requirements.txt
```

### 3. **Verify dataset availability**
All JSONL files should already exist in `datasets/`.
If adding new data, ensure format matches existing split files.

---

## 📘 How It Works
This project consists of a **two-stage QA pipeline**:
1. **Extractive Model** – selects the best candidate answer span.
2. **Refinement Model** – improves the extractive output using a generative model.

Both models can be trained independently and used together.

---

## 🧠 Training
### 🔹 Train the Extractive Model
```bash
python src/train_extractive.py 
```

### 🔹 Train the Refinement Model
```bash
python src/train_refinement.py
```

---

## 🔍 Running Predictions
### Extractive Demo
```bash
python src/run_predictor_demo_extractive.py 
```

### Refinement Demo
```bash
python src/run_predictor_demo_refinement.py 
```

---

## 📝 Development Notes
- `extractive_predictor.py` implements the extractive QA model wrapper.
- `utils.py` includes dataset loading, metric functions, and helpers.
- Ensure consistent tokenization across extractive and refinement stages.
- Training scripts support GPU acceleration if available.

---

## 📦 Reproducibility Checklist
- [ ] Python 3.10+
- [ ] Install `requirements.txt`
- [ ] Ensure datasets are present
- [ ] Run extractive training
- [ ] Run refinement training
- [ ] Generate predictions
- [ ] Results are stored in the form of json

---

## 🤝 Contributing
Feel free to open issues or submit pull requests if you’d like to improve the pipeline or add new evaluation metrics.

---

