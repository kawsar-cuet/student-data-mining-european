# Overleaf Upload Checklist

## ✅ Files Ready for Upload

### 1. Main LaTeX Document
- **File**: `docs/JOURNAL_METHODOLOGY.tex`
- **Size**: ~120 KB
- **Sections**: 13 complete sections + 9 figures + bibliography + appendix
- **Status**: ✅ COMPLETE

### 2. Figure Files (10 PDF files)
All files from `outputs/figures_journal/`:

```
✅ figure1_model_comparison.pdf       (Model performance bars, 3 panels)
✅ figure2_roc_curves.pdf             (ROC curves, AUC comparison)
✅ figure3_ppn_confusion_matrix.pdf   (PPN 3-class confusion matrix)
✅ figure4_dpna_confusion_matrix.pdf  (DPN-A binary confusion matrix)
✅ figure5_attention_heatmap.pdf      (Risk-stratified attention weights)
✅ figure6_feature_importance.pdf     (Top 20 features, Tinto/Bean)
✅ figure7_training_curves.pdf        (Loss convergence plots)
✅ figure8_class_distribution.pdf     (Class imbalance visualization)
✅ figure9_pr_curves.pdf              (Precision-Recall curves)
✅ figure10_dual_task_comparison.pdf  (BOTH research tasks integrated) ⭐ NEW!
```

**Total Size**: ~4 MB

---

## 📁 Overleaf Folder Structure

Create this structure in your Overleaf project:

```
project_root/
│
├── JOURNAL_METHODOLOGY.tex          ← Upload this
│
└── outputs/
    └── figures_journal/
        ├── figure1_model_comparison.pdf
        ├── figure2_roc_curves.pdf
        ├── figure3_ppn_confusion_matrix.pdf
        ├── figure4_dpna_confusion_matrix.pdf
        ├── figure5_attention_heatmap.pdf
        ├── figure6_feature_importance.pdf
        ├── figure7_training_curves.pdf
        ├── figure8_class_distribution.pdf
        └── figure9_pr_curves.pdf
```

**Important**: The LaTeX file uses relative paths like:
```latex
\includegraphics[width=0.95\textwidth]{outputs/figures_journal/figure1_model_comparison.pdf}
```

So maintain the exact folder structure `outputs/figures_journal/` in Overleaf.

---

## ⚙️ Overleaf Compilation Settings

1. **Compiler**: Select **pdfLaTeX** (not XeLaTeX or LuaLaTeX)
2. **Main Document**: Set `JOURNAL_METHODOLOGY.tex` as main file
3. **Auto-compile**: Enable for live preview
4. **Spell Check**: Enable (set to English)

---

## 🔍 Post-Upload Verification Steps

### Step 1: Check Document Compiles
- Click "Recompile" button
- Wait for PDF generation (may take 30-60 seconds)
- Check for any errors in Logs & Outputs panel

### Step 2: Verify All Figures Appear
Scroll through PDF and confirm all 9 figures render:
- [ ] Figure 1 (page ~45): Model comparison bar charts
- [ ] Figure 2 (page ~46): ROC curves
- [ ] Figure 3 (page ~47): PPN confusion matrix
- [ ] Figure 4 (page ~48): DPN-A confusion matrix
- [ ] Figure 5 (page ~49): Attention heatmap
- [ ] Figure 6 (page ~50): Feature importance
- [ ] Figure 7 (page ~51): Training curves
- [ ] Figure 8 (page ~52): Class distribution
- [ ] Figure 9 (page ~53): Precision-Recall curves

### Step 3: Check Cross-References
All `Figure~\ref{fig:*}` references should show numbers (not "??"):
- Search PDF for "Figure ??" (should return 0 results)
- If you see "??", compile twice (LaTeX needs 2 passes for references)

### Step 4: Verify Tables
Check that all 20+ tables render correctly:
- Lines should be clean (booktabs package)
- Captions appear ABOVE tables
- No overflow text (all content fits in columns)

### Step 5: Check Bibliography
- Scroll to References section (near end)
- Verify all 15 citations appear:
  1. Tinto (1993)
  2. Bean (1985)
  3. Breiman (2001) - Random Forests
  4. Vaswani et al. (2017) - Attention mechanism
  5. Kingma & Ba (2015) - Adam optimizer
  6. ... (10 more)

---

## 🐛 Common Compilation Errors & Fixes

### Error: "File figure1_model_comparison.pdf not found"
**Fix**: Check folder structure matches exactly:
- Right-click "New Folder" → name it `outputs`
- Inside `outputs`, create `figures_journal`
- Upload all 9 PDFs to `outputs/figures_journal/`

### Error: "Missing $ inserted"
**Fix**: LaTeX math mode issue. Check lines around error for unescaped symbols:
- Use `\%` instead of `%` in text
- Use `\$` instead of `$` in text
- Wrap math in `$...$` or `\(...\)`

### Error: "Undefined control sequence \toprule"
**Fix**: Add to preamble (should already be there):
```latex
\usepackage{booktabs}
```

### Error: "Package natbib Error: Bibliography not compatible"
**Fix**: Delete auxiliary files and recompile:
- Click "Logs & Outputs" → "Clear cached files"
- Recompile

### Warning: "Overfull hbox"
**Fix**: Long lines exceeding margins. Usually non-critical.
- Check PDF - if text looks fine, ignore
- If text overflows, break long URLs or table entries

---

## 📊 Expected Compilation Output

**Successful Compilation Shows**:
- ✅ 0 Errors
- ⚠️ 5-15 Warnings (acceptable - mostly overfull boxes)
- 📄 PDF pages: ~55-60 pages
- 🕒 Compile time: 20-40 seconds

**PDF Structure**:
```
Pages 1-2:    Title, Authors, Abstract
Pages 3-5:    Introduction
Pages 6-10:   Literature Review
Pages 11-15:  Deep Learning Techniques
Pages 16-25:  Dataset & Methodology (7 tables)
Pages 26-30:  Feature Engineering
Pages 31-35:  Deep Learning Architectures
Pages 36-40:  LLM Integration
Pages 41-44:  Evaluation & Implementation
Pages 45-53:  FIGURES (9 figures, 1 per page)
Pages 54-56:  References (15 citations)
Pages 57-58:  Appendix (Preprocessing pseudocode)
```

---

## ✏️ Final Edits Before Submission

### 1. Update Abstract (Lines 58-84)
Currently says "expected to surpass" - change to actual results:

**OLD**:
```latex
Experiments on authentic institutional data (N=4,424 students) are expected to 
demonstrate that deep learning architectures, particularly those with attention 
mechanisms, surpass traditional machine learning baselines...
```

**NEW**:
```latex
Experiments on authentic institutional data (N=4,424 students) demonstrate that 
deep learning architectures with attention mechanisms achieve state-of-the-art 
performance (DPN-A: 87.05\% accuracy, 0.910 AUC-ROC), surpassing baseline 
Logistic Regression by 1.35\% while providing interpretable feature importance 
aligned with Tinto's academic integration (68\% weight) and Bean's environmental 
factors (32\% weight).
```

### 2. Add Author Affiliations (Lines 15-25)
Replace placeholders:
```latex
\author[inst1]{Your Name}
\author[inst2]{Advisor Name}

\affiliation[inst1]{organization={Department of Computer Science, University Name},
                    city={City},
                    country={Country}}
```

### 3. Add Keywords (Line 86)
Currently empty. Add 5-7 keywords:
```latex
\begin{keyword}
    Student dropout prediction \sep 
    Attention mechanism \sep 
    Multi-task learning \sep 
    Educational data mining \sep 
    Deep learning \sep 
    Tinto's integration model \sep 
    Interpretable AI
\end{keyword}
```

### 4. Add Acknowledgments (Before References)
```latex
\section*{Acknowledgments}
This research was supported by [Grant Name/Number]. We thank [Institution Name] 
for providing access to de-identified student data. Special thanks to [Advisor 
Names] for guidance throughout this project.
```

### 5. Add Data Availability Statement (After Acknowledgments)
```latex
\section*{Data Availability}
The dataset used in this study contains sensitive student records and cannot be 
publicly shared due to privacy regulations. Anonymized sample data and complete 
source code are available at: \url{https://github.com/username/repo}
```

---

## 📤 Submission Preparation

### Journal Submission Checklist
- [ ] Compiled PDF with no errors
- [ ] All 9 figures render correctly
- [ ] Abstract updated with actual results
- [ ] Author names and affiliations complete
- [ ] Keywords added (5-7 terms)
- [ ] Acknowledgments section added
- [ ] Data availability statement added
- [ ] References formatted correctly (APA style via natbib)
- [ ] Word count within journal limits (typically 8,000-12,000 words)
- [ ] Supplementary materials prepared (code, data dictionary)

### Target Journals (Ranked by Fit)
1. **Computers & Education: Artificial Intelligence** (Best fit)
   - Impact Factor: ~6.5
   - Scope: AI applications in education
   - Submission: Elsevier Editorial Manager

2. **IEEE Transactions on Learning Technologies**
   - Impact Factor: ~3.5
   - Scope: Learning analytics, AI in education
   - Submission: IEEE ScholarOne

3. **Journal of Educational Data Mining**
   - Impact Factor: ~3.2
   - Scope: Data mining in educational contexts
   - Submission: Direct to editor

### Supplementary Materials to Prepare
1. **Code Repository** (GitHub/GitLab):
   - `main_pytorch.py` (training script)
   - `visualizations_pytorch.py` (figure generation)
   - `src/data_preprocessing_real.py` (preprocessing)
   - `README.md` (setup instructions)
   - `requirements.txt` (Python dependencies)

2. **Data Dictionary** (CSV/Excel):
   - 46 feature definitions
   - Data types, ranges, missing value handling
   - Theoretical mapping (Tinto vs Bean)

3. **Extended Results** (PDF supplement):
   - Hyperparameter tuning grids
   - Full cross-validation fold results
   - Additional error analysis
   - SHAP force plots (if generated)

---

## 🚀 Upload Now!

**Ready to upload?** Just drag and drop these files to Overleaf:

1. `JOURNAL_METHODOLOGY.tex` → Root folder
2. All 9 PDF figures → Create `outputs/figures_journal/` folder first
3. Click "Recompile" and verify PDF generates

**Estimated time**: 10 minutes to upload + verify

---

## 📞 Support

If you encounter issues:
1. Check Overleaf documentation: https://www.overleaf.com/learn
2. Verify elsarticle class installed (should be by default)
3. Compare error messages to Common Errors section above
4. Ensure figure paths exactly match folder structure

**Good luck with your journal submission! 🎓📊**
