# Quick Start Guide

## Student Performance Prediction System - Quick Setup

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- ~500MB free disk space

---

## Step-by-Step Setup (5 minutes)

### 1. Navigate to Project Directory
```powershell
cd "d:\MS program\Final Thesis\Final Thesis project"
```

### 2. Create Virtual Environment
```powershell
python -m venv venv
```

### 3. Activate Virtual Environment
```powershell
.\venv\Scripts\Activate.ps1
```

If you get an execution policy error, run:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 4. Upgrade pip (recommended)
```powershell
python -m pip install --upgrade pip
```

### 5. Install Dependencies
```powershell
pip install -r requirements.txt
```

This will take 2-3 minutes.

### 6. Set Up Environment Variables (Optional for LLM)
```powershell
Copy-Item .env.example .env
```

Then edit `.env` and add your OpenAI API key if you want LLM-powered recommendations:
```
OPENAI_API_KEY=sk-your-key-here
```

**Note**: The system works without an API key (uses rule-based recommendations instead).

### 7. Run the Complete Pipeline
```powershell
python main.py
```

This will:
- Load and preprocess data
- Train 3 deep learning models
- Evaluate models
- Generate visualizations
- Create personalized recommendations

Expected runtime: 5-10 minutes

---

## Project Structure Overview

```
Final Thesis project/
├── data/
│   └── ulab_students_dataset.csv    # 50 student records
│
├── src/
│   ├── data_preprocessing.py        # Data pipeline
│   ├── models/                      # Deep learning models
│   ├── llm/                         # LLM recommendations
│   ├── evaluation.py                # Metrics
│   └── visualization.py             # Plots
│
├── outputs/
│   ├── models/                      # Saved models (.h5 files)
│   ├── plots/                       # Visualizations (.png)
│   └── reports/                     # Student recommendations (.txt)
│
├── docs/
│   └── METHODOLOGY.md               # Research methodology
│
├── main.py                          # Main execution script
├── requirements.txt                 # Dependencies
└── README.md                        # Full documentation
```

---

## After Running

### Check Results

1. **Model Performance**
   - Check console output for accuracy, precision, recall, F1, AUC

2. **Visualizations**
   - Navigate to `outputs/plots/`
   - View confusion matrices, training history, ROC curves

3. **Recommendations**
   - Navigate to `outputs/reports/`
   - Read personalized recommendations for high-risk students

4. **Saved Models**
   - Located in `outputs/models/`
   - Can be loaded later for predictions

---

## Common Issues

### Issue: "No module named 'tensorflow'"
**Solution**: Make sure virtual environment is activated and run:
```powershell
pip install tensorflow
```

### Issue: "ModuleNotFoundError"
**Solution**: Install missing package:
```powershell
pip install <package-name>
```

### Issue: Out of memory during training
**Solution**: Reduce batch size in `main.py` (line ~70):
```python
batch_size=8  # instead of 16
```

### Issue: Training too slow
**Solution**: Reduce epochs in `main.py` (line ~69):
```python
epochs=50  # instead of 100
```

---

## Next Steps

### Explore the Code
1. Read `docs/METHODOLOGY.md` for research details
2. Review `src/data_preprocessing.py` to understand feature engineering
3. Check `src/models/` to see model architectures

### Modify the Project
1. Add more students to `data/ulab_students_dataset.csv`
2. Adjust model hyperparameters in model files
3. Customize recommendation prompts in `src/llm/recommendation_engine.py`

### Run Jupyter Notebooks
```powershell
jupyter notebook
```

Then open notebooks in `notebooks/` directory for interactive exploration.

---

## Testing Individual Components

### Test Data Preprocessing Only
```powershell
python src/data_preprocessing.py
```

### Test LLM Recommendations Only
```powershell
python src/llm/recommendation_engine.py
```

### Test Visualization Only
```powershell
python src/visualization.py
```

---

## Deactivate Virtual Environment
When done:
```powershell
deactivate
```

---

## Need Help?

1. Check `README.md` for detailed documentation
2. Review `docs/METHODOLOGY.md` for research context
3. Check error messages in console output
4. Ensure all dependencies are installed

---

## Quick Commands Cheat Sheet

```powershell
# Activate environment
.\venv\Scripts\Activate.ps1

# Run main pipeline
python main.py

# Install new package
pip install package-name

# Check installed packages
pip list

# Deactivate environment
deactivate
```

---

**Estimated Total Setup Time**: 5-10 minutes
**First Run Time**: 5-10 minutes
**Subsequent Runs**: 3-5 minutes

Good luck with your research! 🎓
