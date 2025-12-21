# IEEE Journal Paper: AHFS-TA for Student Dropout Prediction

## Overview

This folder contains a high-quality IEEE format journal paper (**15 pages, two-column format**) presenting the AHFS-TA (Adaptive Hierarchical Feature Selection with Temporal Attention) framework for student dropout prediction.

## Files

- `AHFS_TA_Journal_Paper.tex` - Main LaTeX source file (IEEE format, two-column)
- `AHFS_TA_Journal_Paper.pdf` - **Compiled PDF (15 pages, 3.15 MB)**
- `IEEEtran.cls` - IEEE Transactions LaTeX class file
- `figures/` - Key figures for the paper (14 figures)
- `README.md` - This file

## Paper Structure (IEEE Format)

The paper follows IEEE Transactions format with the following sections:

1. **Abstract** - Comprehensive summary with key results
2. **Introduction** - Problem statement, challenges, contributions
3. **Related Work** - Extensive literature review on dropout prediction
4. **Dataset Description** - 4,424 students, 46 features, 3 classes
5. **Feature Ranking Methodology** - 5 methods with meta-ranking
6. **Baseline Models** - 6 models with algorithms (DT, NB, RF, AdaBoost, XGBoost, NN)
7. **Proposed AHFS-TA Framework** - Novel architecture with 4 algorithms
8. **Experimental Setup** - Implementation details, hyperparameters, metrics
9. **Results and Analysis** - Comprehensive comparison tables, figures
10. **Explainable AI Analysis** - SHAP and attention visualization
11. **Discussion** - Practical implications, limitations, future directions
12. **Conclusion** - Summary and future work
13. **References** - 16 key citations

## Key Results

| Model | Accuracy | AUC-ROC | CV Mean ± Std |
|-------|----------|---------|---------------|
| Decision Tree | 67.01% | 0.758 | 67.47% ± 1.30% |
| Naive Bayes | 70.85% | 0.843 | 72.47% ± 2.07% |
| Random Forest | 76.72% | 0.914 | 77.22% ± 1.24% |
| AdaBoost | 74.24% | 0.890 | 74.39% ± 1.17% |
| XGBoost | 75.93% | 0.913 | 78.21% ± 0.81% |
| Neural Network | 71.41% | 0.861 | 72.33% ± 1.49% |
| **AHFS-TA (Ours)** | **91.32%** | **0.955** | **90.85% ± 0.92%** |

## Novel Contributions

1. **AHFS-TA Framework**: Integrates LLM-based psychosocial features, BiGRU with multi-head attention, and three-stream adaptive feature selection

2. **Multi-Method Feature Ranking**: Combines 5 techniques (Information Gain, Gain Ratio, Gini Index, Chi-squared, ANOVA F-statistic)

3. **Comprehensive SHAP Analysis**: Explainability for all 7 models

4. **Temporal Attention Insights**: Reveals Semesters 2-3 as critical intervention periods

## Compilation

The paper is pre-compiled. To recompile:

```bash
cd "Journal Paper Writing"
pdflatex AHFS_TA_Journal_Paper.tex
pdflatex AHFS_TA_Journal_Paper.tex
```

Or upload all files to Overleaf.

## Supervisor Requirements Addressed

✅ Total students: 4,424
✅ Total features: 46 (Academic 18, Financial 12, Demographic 16)
✅ Classes: 3 (Dropout 1,421, Enrolled 794, Graduate 2,209)
✅ Feature lists with names
✅ Feature rankings (5 methods)
✅ All 6 baseline models trained
✅ All metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC, AUC
✅ 10-Fold Cross-Validation with mean ± std
✅ Explainable AI (SHAP) for all models
✅ Novel AHFS-TA unique algorithm

## Paper Length

- **Pages: 15** (IEEE two-column format)
- **File Size: 3.15 MB** (with embedded figures)
- Tables: 14
- Figures: 8 (embedded in PDF)
- Algorithms: 4
- References: 16
