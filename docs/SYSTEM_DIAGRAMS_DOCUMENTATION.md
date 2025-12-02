# System Architecture Diagrams - Complete Documentation

## ✅ Generated: 3 Professional System Design Diagrams

**Date**: November 30, 2025  
**Output**: `outputs/figures_journal/`  
**Format**: PDF (vector) + PNG (raster), 300 DPI

---

## 📊 Diagram 1: Complete End-to-End System Architecture

**File**: `system_architecture_complete.pdf` (and .png)  
**Size**: 18" × 14" (publication quality)

### Overview
Full system workflow from raw data to deployment, organized in 6 hierarchical layers.

### Layer Structure

#### **Layer 1: Data Sources** (Top)
- **Data Sources**: Educational dataset (N=4,424 students)
- **Raw Features**: 37 variables (Academic | Financial | Demographic)
- **Target Variables**: Performance (3-class), Dropout (Binary)
- **Theoretical Framework**: Tinto (68%), Bean (32%)

#### **Layer 2: Preprocessing Pipeline**
7 preprocessing stages:
1. Missing Value Imputation (KNN)
2. Categorical Encoding (One-Hot, Label)
3. Feature Engineering (Success rate, Average grade, etc.)
4. Normalization (Z-score standardization)
5. Class Weight Calculation (Handle imbalance)
6. Train/Val/Test Split (70/15/15)
7. Tensor Conversion (PyTorch)

**Output**: Train: 3,096 | Val: 664 | Test: 664

#### **Layer 3: Model Architectures** (3 parallel models)

**PPN (Performance Prediction Network)**:
- Input: 46 features
- FC1: 46 → 128 (ReLU, BatchNorm, Dropout 0.3)
- FC2: 128 → 64 (ReLU, BatchNorm, Dropout 0.2)
- FC3: 64 → 32 (ReLU, Dropout 0.1)
- Output: 32 → 3 (Softmax)

**DPN-A (Dropout Prediction with Attention)**:
- Input: 46 features
- FC1: 46 → 64 (ReLU, BatchNorm, Dropout 0.3)
- **🔍 Self-Attention Layer** (64-dim)
- FC2: 64 → 32 (ReLU, Dropout 0.2)
- FC3: 32 → 16 (ReLU)
- Output: 16 → 1 (Sigmoid)

**HMTL (Hybrid Multi-Task Learning)**:
- Shared trunk: 46 → 128 → 64
- Head 1 (Performance): 64 → 32 → 3
- Head 2 (Dropout): 64 → 16 → 1
- Combined loss: L_total = L_perf + λL_drop (λ=1.0)

#### **Layer 4: Training & Optimization**
- **Optimizer**: Adam (learning rate = 0.001)
- **Loss Functions**: 
  - CrossEntropy (PPN, HMTL-Performance)
  - Binary CrossEntropy (DPN-A, HMTL-Dropout)
- **Early Stopping**: Patience = 20 epochs
- **LR Scheduler**: ReduceLROnPlateau (factor=0.5, patience=10)

#### **Layer 5: Evaluation Metrics**
8 comprehensive metrics:
1. Accuracy
2. F1-Macro
3. Precision
4. Recall
5. AUC-ROC
6. AUC-PR
7. Confusion Matrix
8. 10-Fold Cross-Validation

#### **Layer 6: Results & Deployment** (Bottom)

**PPN Results**:
- Accuracy: 76.4%
- F1-Macro: 0.688

**DPN-A Results** ⭐ **BEST**:
- Accuracy: 87.05%
- AUC-ROC: 0.910
- F1-Score: 0.782

**HMTL Results**:
- Performance task: 76.4%
- Dropout task: 67.9% ⚠ (task interference)

**Deployment**: Institutional Early Warning System | Intervention Recommendations

### Visual Design
- **Color Coding**:
  - Light Blue: Data layer
  - Light Orange: Preprocessing
  - Light Green: Model architectures
  - Light Purple: Evaluation
  - Light Yellow: Deployment
- **Flow**: Top-to-bottom with arrows showing data flow
- **Legend**: Color-coded layers for easy interpretation

---

## 🏗️ Diagram 2: Detailed Model Architectures Comparison

**File**: `system_architecture_models.pdf` (and .png)  
**Size**: 18" × 10" (3-column layout)

### Overview
Side-by-side detailed comparison of all three deep learning architectures.

### Column 1: PPN Architecture

**Title**: PPN: Performance Prediction Network (3-Class Classification)

**Layer Stack** (vertical flow):
1. **Input Layer**: 46 features
2. **FC1**: 46 → 128
   - ReLU activation
   - BatchNorm
   - Dropout(0.3)
3. **FC2**: 128 → 64
   - ReLU activation
   - BatchNorm
   - Dropout(0.2)
4. **FC3**: 64 → 32
   - ReLU activation
   - Dropout(0.1)
5. **Output**: 32 → 3
   - Softmax activation
6. **Predictions**: [Dropout, Enrolled, Graduate]

**Performance Box**: 
- Accuracy: 76.4%
- F1-Macro: 0.688

**Color Scheme**: Blue gradient (light to dark)

### Column 2: DPN-A Architecture

**Title**: DPN-A: Dropout Prediction with Attention (Binary Classification)

**Layer Stack** (vertical flow):
1. **Input Layer**: 46 features
2. **FC1**: 46 → 64
   - ReLU activation
   - BatchNorm
   - Dropout(0.3)
3. **🔍 Self-Attention Layer**: 64-dim
   - Q, K, V projection
   - Scaled dot-product attention
   - **Highlighted with annotation box**
4. **FC2**: 64 → 32
   - ReLU activation
   - Dropout(0.2)
5. **FC3**: 32 → 16
   - ReLU activation
6. **Output**: 16 → 1
   - Sigmoid activation
7. **Prediction**: [Dropout Probability]

**Performance Box** (highlighted):
- Accuracy: 87.05%
- AUC-ROC: 0.910
- F1-Score: 0.782

**Color Scheme**: Orange gradient (light to dark)  
**Special Feature**: Attention layer emphasized with arrow annotation

### Column 3: HMTL Architecture

**Title**: HMTL: Hybrid Multi-Task Learning (Dual Outputs)

**Shared Trunk** (vertical):
1. **Input Layer**: 46 features
2. **Shared FC1**: 46 → 128 (ReLU)
3. **Shared FC2**: 128 → 64 (ReLU)

**Split into Two Heads** (branching arrows):

**Left Branch - Performance Head**:
- Head 1: 64 → 32
- Output: 32 → 3 (Softmax)
- Performance Output
- Result: **Acc: 76.4%**

**Right Branch - Dropout Head**:
- Head 2: 64 → 16
- Output: 16 → 1 (Sigmoid)
- Dropout Output
- Result: **Acc: 67.9% ⚠**

**Loss Function Box** (bottom center):
- **Combined Loss**: L_total = L_perf + λL_drop
- λ = 1.0 (equal weighting)

**Warning Box**: ⚠ Task Interference: Dropout task underperforms (-19.15% vs DPN-A)

**Color Scheme**: 
- Purple gradient (shared trunk)
- Green (performance head)
- Red (dropout head)

### Visual Features
- **Vertical flow**: Top-to-bottom layer stacking
- **Arrows**: Show data flow between layers
- **Color gradients**: Indicate depth progression
- **Annotations**: Highlight key features (attention, task interference)
- **Performance boxes**: Summary metrics at bottom

---

## 🔄 Diagram 3: Data Flow & Feature Engineering Pipeline

**File**: `system_architecture_dataflow.pdf` (and .png)  
**Size**: 16" × 10"

### Overview
Detailed visualization of data preprocessing and feature engineering workflow.

### Horizontal Flow (Left to Right)

#### **Stage 1: Raw Data**
- **Source**: CSV file
- **Size**: 4,424 rows × 37 columns
- **Color**: Light blue

#### **Stage 2: Data Cleaning**
- ✓ Remove duplicates
- ✓ Handle missing values (KNN imputation)
- ✓ Drop invalid rows
- **Color**: Light orange

#### **Stage 3: Encoding**
- Categorical → Numeric conversion
- One-Hot encoding: gender, marital status
- Label encoding: daytime/evening attendance
- **Color**: Light green

#### **Stage 4: Feature Engineering**
- **Added features**:
  - + Success rate
  - + Average grade
  - + Academic progression
- **Color**: Light purple

### Vertical Flow (Top to Bottom)

#### **Feature Categorization** (3 parallel boxes)

**Academic Features** (Blue box):
- Grades (semester 1, 2)
- Success rate
- Enrolled units
- Approved units
- Evaluations

**Financial Features** (Orange box):
- Tuition status
- Scholarship holder
- Debtor status
- Payment history

**Demographic Features** (Green box):
- Age at enrollment
- Gender
- Marital status
- Parental education level
- Previous qualification

#### **Normalization Stage**
- **Method**: Z-Score standardization
- **Formula**: x_norm = (x - μ) / σ
- **Applied to**: All 46 features
- **Color**: Light yellow

#### **Final Output** (Bottom)
- **Processed Dataset**: 46 Features × 4,424 Samples
- **Data Split**:
  - Train: 3,096 (70%)
  - Validation: 664 (15%)
  - Test: 664 (15%)
- **Color**: Light purple (emphasized)

### Theoretical Mapping (Side Panel)

**Theory Integration Box** (Left side):
- **Tinto's Model**: 68% weight (blue)
  - Maps to Academic features
- **Bean's Model**: 32% weight (orange)
  - Maps to Financial features
- **Dashed arrows**: Connect theories to feature categories

### Visual Features
- **Horizontal arrows**: Stage-to-stage progression
- **Vertical arrows**: Top-down refinement
- **Color coding**: Different colors for each stage
- **Dotted lines**: Theory-to-feature mapping
- **Box hierarchy**: Nested boxes show relationships

---

## 📐 Technical Specifications

### All Diagrams

**Resolution**: 300 DPI (publication quality)  
**Format**: 
- PDF: Vector graphics (lossless scaling)
- PNG: Raster graphics (preview/presentation)

**Software**: 
- Matplotlib 3.x
- Custom FancyBboxPatch and FancyArrowPatch

**Typography**:
- Title: 16-18pt, bold
- Section headers: 10-12pt, bold
- Body text: 7-9pt, regular
- Monospace: Code/formulas

**Color Palette**:
- Consistent across all diagrams
- Colorblind-friendly selections
- Semantic color coding (e.g., green=success, red=warning)

**Layout**:
- Professional spacing
- Clear visual hierarchy
- Aligned components
- Consistent padding

---

## 🎯 Usage Recommendations

### For Thesis Defense Presentation
1. **Use Diagram 1** in "System Overview" slide
2. **Use Diagram 2** in "Model Architecture" slide
3. **Use Diagram 3** in "Data Pipeline" slide

### For Journal Manuscript
1. **Diagram 1**: Main text Figure (after methodology section)
2. **Diagram 2**: Supplementary material (detailed architectures)
3. **Diagram 3**: Main text Figure (in data preprocessing section)

### For Technical Documentation
- All 3 diagrams in separate sections
- Use PNG for web/README
- Use PDF for printed documentation

---

## 📊 Comparison with Previous Visualizations

### Before (Figures 1-10)
- Focused on **results** (confusion matrices, ROC curves, performance bars)
- Model **outputs** and **evaluations**
- Comparative analysis

### Now (System Diagrams)
- Focused on **architecture** and **design**
- Model **internals** and **data flow**
- System **structure** and **engineering**

### Complementary Coverage
**Results Figures** (1-10): "What did we achieve?"  
**System Diagrams** (3 new): "How did we build it?"

---

## 🚀 Next Steps

### Integration into LaTeX
Add these as supplementary figures or appendix:

```latex
\begin{figure}[p]
    \centering
    \includegraphics[width=\textwidth]{outputs/figures_journal/system_architecture_complete.pdf}
    \caption{\textbf{Complete End-to-End System Architecture.} 
    Six-layer architecture showing data flow from raw educational 
    dataset through preprocessing, three deep learning models 
    (PPN, DPN-A, HMTL), training optimization, evaluation metrics, 
    and deployment for institutional early warning system.}
    \label{fig:system_arch_complete}
\end{figure}
```

### For Presentation Slides
1. Export PNG versions
2. Use as full-slide backgrounds
3. Add incremental reveals for layers

### For README/Documentation
```markdown
## System Architecture

![Complete System](outputs/figures_journal/system_architecture_complete.png)

### Model Architectures
![Models](outputs/figures_journal/system_architecture_models.png)

### Data Pipeline
![Data Flow](outputs/figures_journal/system_architecture_dataflow.png)
```

---

## 📝 Files Generated

| File | Type | Size | Purpose |
|------|------|------|---------|
| `system_architecture_complete.pdf` | Vector | ~50 KB | End-to-end system (6 layers) |
| `system_architecture_complete.png` | Raster | ~850 KB | Preview/presentation |
| `system_architecture_models.pdf` | Vector | ~40 KB | 3 model architectures side-by-side |
| `system_architecture_models.png` | Raster | ~750 KB | Preview/presentation |
| `system_architecture_dataflow.pdf` | Vector | ~45 KB | Feature engineering pipeline |
| `system_architecture_dataflow.png` | Raster | ~800 KB | Preview/presentation |

**Total**: 6 files, ~2.5 MB

---

## ✨ Summary

**3 professional system architecture diagrams** covering:

1. ✅ **Complete end-to-end workflow** (Data → Deployment)
2. ✅ **Detailed model internals** (PPN, DPN-A, HMTL)
3. ✅ **Feature engineering pipeline** (Raw data → Processed tensors)

**Ready for**:
- ✅ Thesis defense presentations
- ✅ Journal manuscript figures
- ✅ Technical documentation
- ✅ GitHub README illustrations
- ✅ Conference posters

**Publication quality**: 300 DPI, vector PDF, professional design, clear typography, semantic color coding!

---

**Your complete research system is now fully visualized and documented!** 🎓🏗️✨
