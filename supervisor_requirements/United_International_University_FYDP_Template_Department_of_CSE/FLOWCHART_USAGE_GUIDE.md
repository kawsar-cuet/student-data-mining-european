# Methodology Flowcharts - Usage Guide

## 📊 Generated Flowcharts

Two professional flowcharts have been created for your thesis:

### 1. **Advanced Flowchart** (`methodology_flowchart_advanced.png`)
- **Size:** 14×18 inches (portrait)
- **Resolution:** 300 DPI (publication quality)
- **Content:** Complete detailed methodology with all phases
- **Includes:**
  - Dataset overview (4,424 students, 46 features)
  - Feature engineering & preprocessing (12 engineered features)
  - Ensemble feature ranking (5 methods)
  - Data partitioning (80/10/10 split)
  - Hyperparameter optimization loop (1,728 configurations)
  - Three model architectures (PPN, DPN-A, HMTL)
  - Comprehensive evaluation & cross-validation
  - Interpretability analysis (Tinto/Bean validation)
  - LLM integration (GPT-4)
  - Phase annotations (Phase 1-8)

### 2. **Simplified Flowchart** (`methodology_flowchart_simple.png`)
- **Size:** 12×14 inches (portrait)
- **Resolution:** 300 DPI (publication quality)
- **Content:** Streamlined overview of methodology
- **Best for:** Quick understanding of research workflow

---

## 📝 How to Insert into Your Thesis

### Option 1: Insert in Chapter 3 (Methodology) - After Section 3.1

Add this LaTeX code after describing your dataset:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/methodology_flowchart_advanced.png}
\caption{\textbf{Comprehensive Research Methodology Flowchart.} Complete workflow showing all phases from data preprocessing through LLM-powered intervention generation. The flowchart illustrates: (1) Feature engineering and preprocessing with 12 derived features, (2) Ensemble feature ranking using 5 methods, (3) Stratified data partitioning, (4) Hyperparameter optimization across 1,728 configurations for three deep learning architectures (PPN, DPN-A, HMTL), (5) Comprehensive evaluation using 10-fold cross-validation and SHAP analysis, (6) Interpretability analysis validating Tinto (68.2\%) and Bean (31.8\%) theoretical frameworks, and (7) GPT-4 integration for personalized recommendations achieving 92\% expert relevance.}
\label{fig:methodology_flowchart}
\end{figure}
```

### Option 2: Insert in Chapter 1 (Introduction) - After Methodology Section

Add this for a high-level overview:

```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.85\textwidth]{figures/methodology_flowchart_simple.png}
\caption{\textbf{Research Methodology Overview.} Simplified flowchart showing the systematic approach from dataset preparation through final results. Key stages include data preprocessing, feature ranking, model training with hyperparameter optimization, evaluation using 10-fold cross-validation, interpretability analysis, and LLM-powered intervention generation.}
\label{fig:methodology_overview}
\end{figure}
```

### Option 3: Use Both Flowcharts

- **Chapter 1 (Introduction):** Use simplified flowchart for overview
- **Chapter 3 (Methodology):** Use advanced flowchart for detailed explanation

---

## 🎨 Flowchart Design Features

### Color Coding (matches your reference image)
- **Light Gray** - Dataset/Input
- **Light Blue** - Processing steps
- **Light Green** - Ensemble/Feature ranking
- **Light Orange** - Decision/Evaluation points
- **Light Yellow** - Optimization loops

### Visual Elements
- ✅ **Rounded boxes** for processes
- ✅ **Arrows** showing workflow direction
- ✅ **Nested boxes** for grouped operations
- ✅ **Phase labels** (Phase 1-8) on the left side
- ✅ **Feedback loops** for iterative processes (red arrows)

### Typography
- **Bold headings** for main sections
- **Smaller text** for details and parameters
- **Professional fonts** matching academic standards

---

## 🔄 Customizing the Flowcharts

If you need to modify the flowcharts, edit the Python script:

```bash
python generate_methodology_flowchart_advanced.py
```

### Common Customizations:

1. **Change colors:**
   - Edit the color variables at the top of the script
   - Example: `color_process = '#YOUR_COLOR'`

2. **Modify text:**
   - Find the relevant `create_box()` or `create_text()` call
   - Update the text parameter

3. **Add/remove sections:**
   - Add new boxes with `create_box()`
   - Connect with `create_arrow()`
   - Adjust y-positions accordingly

4. **Resize:**
   - Change `figsize=(width, height)` in the script
   - Regenerate the images

---

## 📐 Recommended Placement in Thesis

### Best Location: Chapter 3, Section 3.1 (After Dataset Description)

Add the flowchart after describing your dataset characteristics. This provides readers with a visual roadmap before diving into detailed methodology.

**Example integration:**

```latex
\section{Dataset Characteristics and Feature Organization}

% ... your existing dataset description ...

The following flowchart (Figure \ref{fig:methodology_flowchart}) provides a comprehensive overview of the complete research methodology, illustrating the systematic approach from data preprocessing through model evaluation and LLM integration.

\begin{figure}[h]
\centering
\includegraphics[width=0.95\textwidth]{figures/methodology_flowchart_advanced.png}
\caption{... caption as shown above ...}
\label{fig:methodology_flowchart}
\end{figure}

\subsection{Feature Categories}
% Continue with your existing content...
```

---

## 🎯 Key Highlights in the Flowchart

The advanced flowchart emphasizes:

1. **Comprehensive Data Preparation**
   - 12 engineered features
   - Multiple normalization steps
   - Stratified partitioning

2. **Robust Feature Selection**
   - 5 ensemble ranking methods
   - Systematic feature evaluation

3. **Extensive Model Training**
   - 1,728 hyperparameter configurations
   - 3 distinct architectures
   - Adam optimization with early stopping

4. **Rigorous Evaluation**
   - 10-fold cross-validation
   - Multiple metrics (Accuracy, F1, AUC-ROC, etc.)
   - SHAP interpretability

5. **Theoretical Validation**
   - Tinto framework (68.2%)
   - Bean framework (31.8%)
   - Attention weight analysis

6. **Innovation**
   - LLM integration (GPT-4)
   - 92% expert relevance
   - Personalized interventions

---

## 📊 Integration Examples from Your Thesis

### Reference the flowchart in text:

```latex
As illustrated in Figure \ref{fig:methodology_flowchart}, our methodology 
comprises nine distinct phases, systematically progressing from raw data 
to actionable intervention recommendations.

The feature ranking ensemble (Figure \ref{fig:methodology_flowchart}, 
Phase 3) employs five complementary methods to identify the most 
predictive features...

Our hyperparameter optimization loop (Figure \ref{fig:methodology_flowchart}, 
Phase 5) evaluates 1,728 configurations across three model architectures...
```

---

## ✅ Checklist for Using the Flowcharts

- [ ] Choose which flowchart to use (advanced, simple, or both)
- [ ] Add LaTeX code to appropriate chapter file
- [ ] Write descriptive caption (see examples above)
- [ ] Assign unique label (e.g., `\label{fig:methodology_flowchart}`)
- [ ] Reference the figure in surrounding text
- [ ] Recompile LaTeX: `pdflatex fydp.tex` (twice for references)
- [ ] Verify figure appears correctly in PDF
- [ ] Check figure numbering is sequential

---

## 🎨 Comparison with Reference Image

Your generated flowcharts follow the same professional style as the AFSA flowchart you provided:

| Feature | Reference (AFSA) | Your Flowcharts |
|---------|------------------|-----------------|
| **Layout** | Vertical flow, top-to-bottom | ✅ Same |
| **Color coding** | Light blue boxes | ✅ Similar palette |
| **Nested sections** | Feature ranking ensemble | ✅ Same for ranking/training |
| **Loops** | Iterative feature selection | ✅ Optimization loop |
| **Typography** | Bold titles, small details | ✅ Same hierarchy |
| **Arrows** | Clear directional flow | ✅ Same style |
| **Professional quality** | Publication-ready | ✅ 300 DPI, high quality |

---

## 🔧 Troubleshooting

### If flowchart doesn't appear in PDF:

1. **Check file location:**
   ```bash
   ls figures/methodology_flowchart_advanced.png
   ```

2. **Verify LaTeX path:**
   - Use: `figures/methodology_flowchart_advanced.png`
   - NOT: `./figures/...` or `../figures/...`

3. **Recompile twice:**
   ```bash
   pdflatex fydp.tex
   pdflatex fydp.tex
   ```

### If flowchart is too large/small:

Adjust width in LaTeX:
```latex
\includegraphics[width=0.95\textwidth]{...}  % 95% of text width
\includegraphics[width=0.8\textwidth]{...}   % 80% of text width
\includegraphics[width=1.0\textwidth]{...}   % 100% of text width
```

### If you need landscape orientation:

```latex
\begin{sidewaysfigure}
\centering
\includegraphics[width=0.95\textwidth]{figures/methodology_flowchart_advanced.png}
\caption{...}
\label{fig:methodology_flowchart}
\end{sidewaysfigure}
```
(Requires `\usepackage{rotating}` in preamble)

---

## 📚 Summary

You now have two professional, publication-quality flowcharts that:
- ✅ Match the style of your reference AFSA flowchart
- ✅ Comprehensively represent your thesis methodology
- ✅ Are ready to insert into your LaTeX document
- ✅ Include all key research phases and results
- ✅ Follow academic visualization standards

Simply copy the LaTeX code above into your thesis and recompile!

---

**Generated:** December 15, 2025
**Files:** `methodology_flowchart_advanced.png`, `methodology_flowchart_simple.png`
**Location:** `figures/` directory
**Status:** ✅ Ready for thesis integration
