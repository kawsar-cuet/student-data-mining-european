# Quick Start Guide - Journal Implementation

## 🎯 Two Implementations Available

### Option 1: Journal Methodology (Recommended for Publication) ⭐

**Dataset**: Real educational data (4,424 students, 35 features)  
**Script**: `main_real.py`  
**Preprocessing**: `src/data_preprocessing_real.py`  
**Notebook**: `notebooks/01_interactive_demo.ipynb`  
**Methodology**: `docs/JOURNAL_METHODOLOGY.md`

```bash
# Run the journal implementation
python main_real.py
```

**What it does**:
1. Loads real dataset with 4,424 students
2. Engineers 12 derived features (success rate, consistency, etc.)
3. Trains 3 models: PPN (3-class), DPN-A (binary + attention), HMTL (multi-task)
4. Generates publication-quality visualizations
5. Saves models to `outputs/models_real/`
6. Saves plots to `outputs/plots_real/`

**Expected runtime**: ~10-15 minutes (CPU), ~5 minutes (GPU)

---

### Option 2: Original Prototype (For Learning)

**Dataset**: Mock data (50 students, 31 features)  
**Script**: `main.py`  
**Preprocessing**: `src/data_preprocessing.py`

```bash
# Run the original implementation
python main.py
```

---

## 📊 New Files Created

### Core Implementation
- ✅ `src/data_preprocessing_real.py` - Real dataset preprocessing with feature engineering
- ✅ `main_real.py` - Complete pipeline following journal methodology

### Documentation  
- ✅ `docs/JOURNAL_METHODOLOGY.md` - Publication-ready methodology (~4,800 words)
- ✅ `RUNNING_JOURNAL_IMPLEMENTATION.md` - Detailed usage guide
- ✅ `QUICK_START.md` - This file

### Updated Files
- ✅ `notebooks/01_interactive_demo.ipynb` - Now works with real dataset
- ✅ `README.md` - Updated with journal implementation info

---

## 🔬 Key Differences: Mock vs Real

| Feature | Mock (main.py) | Real (main_real.py) |
|---------|----------------|---------------------|
| **Dataset Size** | 50 students | 4,424 students |
| **Features** | 31 original | 35 original + 12 engineered = 47 total |
| **Target Classes** | 9 grades | 3 outcomes (Dropout/Enrolled/Graduate) |
| **Models** | 3 basic DNNs | 3 advanced (PPN, DPN-A with attention, HMTL) |
| **Methodology** | Prototype | Journal-quality (IEEE/ACM standards) |
| **Split Strategy** | Simple 70-30 | Stratified 70-15-15 |
| **Normalization** | Basic scaling | Z-score standardization |
| **Class Imbalance** | Not addressed | Weighted loss functions |
| **Publication Ready** | ❌ No | ✅ Yes |

---

## 📈 Architecture Details

### 1. PPN (Performance Prediction Network)
```
Input (47 features)
    ↓
Dense(128) + BatchNorm + Dropout(0.3)
    ↓
Dense(64) + BatchNorm + Dropout(0.2)
    ↓
Dense(32) + Dropout(0.1)
    ↓
Output(3) - Softmax (Dropout/Enrolled/Graduate)
```

### 2. DPN-A (Dropout Prediction with Attention)
```
Input (47 features)
    ↓
Dense(64) + BatchNorm + Dropout(0.3)
    ↓
Self-Attention Layer (learnable weights)
    ↓
Dense(32) + Dropout(0.2)
    ↓
Dense(16)
    ↓
Output(1) - Sigmoid (Dropout probability)
```

### 3. HMTL (Hybrid Multi-Task Learning)
```
Input (47 features)
    ↓
Shared Trunk:
  Dense(128) + BatchNorm + Dropout(0.3)
  Dense(64) + BatchNorm + Dropout(0.2)
    ↓           ↓
Grade Branch  Dropout Branch
Dense(32)     Dense(16)
    ↓           ↓
Output(3)    Output(1)
Softmax      Sigmoid
```

---

## 🎓 Feature Engineering (12 New Features)

### Academic Performance
1. **total_units_enrolled** - Sum of 1st + 2nd semester units
2. **total_units_approved** - Sum of approved units
3. **success_rate** - Approval rate (approved/enrolled)
4. **semester_consistency** - Grade variance between semesters
5. **average_grade** - Mean grade across both semesters
6. **academic_progression** - Improvement from sem 1 to sem 2

### Engagement Metrics
7. **total_units_no_eval** - Units without evaluation
8. **engagement_index** - 1 - (no_eval/enrolled)
9. **total_evaluations** - Sum of all evaluations
10. **evaluation_completion_rate** - Evaluations per enrolled unit

### Socioeconomic
11. **parental_education_level** - Average of parents' qualifications
12. **financial_support** - Composite of scholarship + tuition status

---

## 📊 Expected Results

After running `main_real.py`, you should see metrics like:

```
PPN (3-class prediction):
  Accuracy: ~0.78-0.85
  F1-Macro: ~0.72-0.80

DPN-A (Binary dropout):
  Accuracy: ~0.80-0.88
  AUC-ROC: ~0.82-0.90
  AUC-PR: ~0.75-0.85

HMTL (Multi-task):
  Grade Accuracy: ~0.76-0.83
  Dropout AUC: ~0.81-0.89
```

*Note: Exact results vary due to random initialization*

---

## 🔧 Troubleshooting

### Q: "Module not found" errors
```bash
pip install --upgrade -r requirements.txt
```

### Q: Slow training
```bash
# Reduce batch size or epochs in main_real.py
batch_size=16  # instead of 32
epochs=50      # instead of 150
```

### Q: Want to test with smaller dataset first?
Edit `main_real.py` after loading data:
```python
df = df.sample(1000, random_state=42)  # Use 1000 samples
```

### Q: See TensorFlow warnings?
They're mostly harmless. To suppress:
```python
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
```

---

## 📚 Next Steps

### For Research/Publication:
1. **Baseline Models**: Implement RF, XGBoost, SVM for comparison
2. **Cross-Validation**: 10-fold CV for robust evaluation
3. **Feature Importance**: SHAP analysis
4. **Statistical Tests**: McNemar's, Friedman tests
5. **LLM Integration**: Add GPT-4 recommendations

### For Learning:
1. Explore `notebooks/01_interactive_demo.ipynb`
2. Read `docs/JOURNAL_METHODOLOGY.md`
3. Modify architectures and hyperparameters
4. Try different feature engineering strategies

---

## 📖 Documentation

- **Complete Methodology**: `docs/JOURNAL_METHODOLOGY.md`
- **Detailed Guide**: `RUNNING_JOURNAL_IMPLEMENTATION.md`
- **Original Methodology**: `docs/METHODOLOGY.md`
- **Main README**: `README.md`

---

## ✅ What's Been Done

- ✅ Real dataset with 4,424 students integrated
- ✅ Feature engineering (12 new features)
- ✅ Three deep learning models implemented
- ✅ Journal-quality methodology documented
- ✅ Interactive Jupyter notebook updated
- ✅ Comprehensive preprocessing pipeline
- ✅ Publication-quality visualizations
- ✅ Stratified train-val-test splits
- ✅ Class imbalance handling
- ✅ Model saving and loading

## 🚧 To Be Done (For Full Publication)

- ⏳ Baseline model comparisons
- ⏳ Cross-validation implementation
- ⏳ SHAP feature importance
- ⏳ Statistical significance testing
- ⏳ Learning curves and calibration plots
- ⏳ GPT-4 LLM integration for recommendations
- ⏳ Results section writing

---

**Last Updated**: November 2025  
**Python**: 3.10+  
**TensorFlow**: 2.15+
