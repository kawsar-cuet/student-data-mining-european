# Running the Journal Methodology Implementation

This document explains how to run the journal-quality implementation with the real educational dataset.

## Overview

The project now includes TWO implementations:

### 1. **Original Implementation** (Mock Dataset)
- Files: `main.py`, `src/data_preprocessing.py`
- Dataset: `data/ulab_students_dataset.csv` (50 students, 31 features)
- Purpose: Prototype and learning

### 2. **Journal Methodology Implementation** (Real Dataset) ⭐
- Files: `main_real.py`, `src/data_preprocessing_real.py`
- Dataset: `data/educational_data.csv` (4,424 students, 35 features)
- Purpose: Publication-ready research

---

## Quick Start - Journal Implementation

### Step 1: Ensure Dataset Exists

Make sure you have the real dataset:
```
data/educational_data.csv
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Full Pipeline

```bash
python main_real.py
```

---

## What the Pipeline Does

### Phase 1: Data Preprocessing
- Loads 4,424 student records with 35 features
- Engineers 12 derived features (success rate, semester consistency, etc.)
- Performs stratified 70-15-15 train-validation-test split
- Applies Z-score normalization
- Output: Ready-to-train datasets

### Phase 2: Model Training
Trains three deep learning models following journal methodology:

1. **PPN (Performance Prediction Network)**
   - Architecture: 128 → 64 → 32 neurons
   - Task: 3-class outcome prediction (Dropout/Enrolled/Graduate)
   - Regularization: BatchNorm + Dropout (0.3, 0.2, 0.1)

2. **DPN-A (Dropout Prediction with Attention)**
   - Architecture: 64 → Attention → 32 → 16 neurons
   - Task: Binary dropout prediction
   - Special: Self-attention layer for interpretability

3. **HMTL (Hybrid Multi-Task Learning)**
   - Architecture: Shared trunk (128 → 64) + dual heads
   - Task: Joint grade and dropout prediction
   - Advantage: Shared representations

### Phase 3: Model Evaluation
- Comprehensive metrics (Accuracy, F1-macro, F1-weighted, AUC-ROC, AUC-PR)
- Classification reports for all models
- Performance comparison tables

### Phase 4: Visualization
Generates publication-quality plots:
- Confusion matrices for all models
- ROC curves comparing models
- Model performance comparison charts
- Saved to: `outputs/plots_real/`

### Phase 5: Model Persistence
Saves trained models:
- `outputs/models_real/ppn_model.h5`
- `outputs/models_real/dpn_attention_model.h5`
- `outputs/models_real/hmtl_model.h5`

---

## Expected Output

```
================================================================================
  STUDENT PERFORMANCE AND DROPOUT PREDICTION SYSTEM
================================================================================

Journal Methodology Implementation - Real Educational Dataset
Dataset: 4,424 Students | Features: 35 | Target: 3-class outcome prediction
Publication Target: IEEE Transactions on Learning Technologies

================================================================================
  PHASE 1: DATA PREPROCESSING
================================================================================

✓ Dataset loaded successfully
  Shape: (4424, 35) (rows × columns)
  Features: 34 + 1 target

  Target Distribution:
    • Graduate: 2209 (49.9%)
    • Dropout: 1421 (32.1%)
    • Enrolled: 794 (18.0%)

... (detailed preprocessing output)

================================================================================
  PHASE 2: DEEP LEARNING MODEL TRAINING
================================================================================

----------------------------------------------------------------------------------
 MODEL 1: PERFORMANCE PREDICTION NETWORK (PPN)
----------------------------------------------------------------------------------

✓ PPN Architecture:
Model: "PPN"
... (model architecture)

🚀 Training PPN (3-class classification)...
  ✓ Training complete
    Best epoch: 68/150
    Best val_loss: 0.4532

... (training continues for DPN-A and HMTL)

================================================================================
  PHASE 3: COMPREHENSIVE MODEL EVALUATION
================================================================================

📊 Overall Metrics:
  Accuracy:        0.8XXX
  F1-Macro:        0.7XXX
  F1-Weighted:     0.8XXX

... (detailed metrics for all models)

================================================================================
  EXECUTION SUMMARY
================================================================================

✓ Data Processing: Complete
  - Total samples: 4,424
  - Features: 47 (including 12 engineered)
  - Train/Val/Test: 3097/664/663

✓ Model Training: Complete
  - PPN:   0.8XXX accuracy, 0.7XXX F1-macro
  - DPN-A: 0.8XXX accuracy, 0.8XXX AUC-ROC
  - HMTL:  0.8XXX grade acc, 0.8XXX dropout AUC

✓ Visualizations: Complete
  - Saved to: outputs/plots_real/

================================================================================
JOURNAL METHODOLOGY IMPLEMENTATION COMPLETED SUCCESSFULLY!
================================================================================
```

---

## Output Files

After running `main_real.py`, you'll find:

### Models (outputs/models_real/)
- `ppn_model.h5` - Performance prediction model
- `dpn_attention_model.h5` - Dropout model with attention
- `hmtl_model.h5` - Multi-task learning model

### Visualizations (outputs/plots_real/)
- `confusion_matrices.png` - 3 confusion matrices side-by-side
- `roc_curves.png` - ROC curves comparing DPN-A and HMTL
- `model_comparison.png` - Bar chart comparing all models

---

## Notebook Demo

To interactively explore the methodology:

```bash
jupyter notebook notebooks/01_interactive_demo.ipynb
```

The notebook has been updated to work with the real dataset and includes:
- Comprehensive data exploration
- Step-by-step preprocessing
- Model training with visual feedback
- Detailed evaluation with plots
- Risk analysis and recommendations

---

## Next Steps for Publication

After running the pipeline, consider:

1. **Baseline Models**: Implement Random Forest, XGBoost, SVM
2. **Cross-Validation**: 10-fold stratified CV for robust metrics
3. **Feature Importance**: SHAP analysis and permutation importance
4. **Statistical Testing**: McNemar's test, Friedman test
5. **LLM Integration**: Add GPT-4 for personalized recommendations
6. **Results Section**: Write up findings following journal format

See `docs/JOURNAL_METHODOLOGY.md` for complete methodology details.

---

## Troubleshooting

### Issue: "FileNotFoundError: data/educational_data.csv"
**Solution**: Ensure the real dataset is in the `data/` folder

### Issue: "Out of memory" during training
**Solution**: Reduce batch size in `main_real.py`:
```python
epochs=150, batch_size=16  # Change from 32 to 16
```

### Issue: Slow training on CPU
**Solution**: 
- Install TensorFlow GPU version
- Or reduce dataset size for testing:
```python
df = df.sample(1000, random_state=42)  # Use 1000 samples for testing
```

### Issue: Import errors
**Solution**: Reinstall dependencies:
```bash
pip install --upgrade -r requirements.txt
```

---

## Configuration

You can modify hyperparameters in `main_real.py`:

```python
# Training configuration
epochs = 150           # Maximum epochs
batch_size = 32       # Batch size
patience = 20         # Early stopping patience
learning_rate = 0.001 # Initial learning rate

# Architecture (modify build_ppn_model, etc.)
hidden_units = [128, 64, 32]  # Layer sizes
dropout_rates = [0.3, 0.2, 0.1]  # Dropout rates
```

---

## Comparison: Mock vs Real Implementation

| Aspect | Mock (main.py) | Real (main_real.py) |
|--------|----------------|---------------------|
| Dataset | 50 students | 4,424 students |
| Features | 31 original | 35 original + 12 engineered |
| Target | 9 grades + binary | 3-class + binary |
| Purpose | Prototype | Publication |
| Methodology | Basic | Journal-quality |
| Documentation | README | Full methodology doc |

---

## Citation

If you use this implementation, please cite:

```
[Your paper title]
[Your name et al.]
[Conference/Journal name]
[Year]
```

---

## Support

For questions or issues:
1. Check `docs/JOURNAL_METHODOLOGY.md` for methodology details
2. Review `notebooks/01_interactive_demo.ipynb` for examples
3. Examine output logs for error messages
4. Verify dataset format matches expected structure

---

**Last Updated**: November 2025  
**Compatible with**: TensorFlow 2.15+, Python 3.10+
