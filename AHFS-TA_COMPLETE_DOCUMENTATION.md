# AHFS-TA: Adaptive Hierarchical Feature Selection with Temporal Attention
## Complete Algorithm Explanation with Examples

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Problem Statement](#problem-statement)
3. [AHFS-TA Framework Overview](#ahfs-ta-framework-overview)
4. [Component 1: LLM-Based Semantic Feature Extraction](#component-1-llm-based-semantic-feature-extraction)
5. [Component 2: Temporal Attention Network](#component-2-temporal-attention-network)
6. [Component 3: Adaptive Hierarchical Feature Selection](#component-3-adaptive-hierarchical-feature-selection)
7. [Component 4: Classification Head](#component-4-classification-head)
8. [Algorithm 2: Complete Training Procedure](#algorithm-2-complete-training-procedure)
9. [End-to-End Example with Real Data](#end-to-end-example-with-real-data)
10. [Why AHFS-TA Works Better](#why-ahfs-ta-works-better)

---

## Executive Summary

**AHFS-TA** is a novel deep learning framework for **student dropout prediction** that addresses three critical challenges:

1. **Temporal Dynamics**: Traditional models treat student records as static snapshots. AHFS-TA captures how behavior changes across semesters using BiGRU + Multi-Head Attention.

2. **Feature Interpretability**: Instead of using all features equally, AHFS-TA adaptively selects only the 28 most important features using a three-stream consensus approach (SHAP + Correlation + Temporal Variance).

3. **Semantic Understanding**: AHFS-TA enriches raw academic data with LLM-derived psychosocial features (engagement, sentiment, topic consistency) using DistilBERT.

**Key Performance**: 91.32% accuracy, 95.5% AUC-ROC (vs. baseline models: 67-76%)

---

## Problem Statement

### Why Student Dropout Prediction is Challenging

**Scenario**: Two students have identical cumulative GPA of 3.3

```
Student A (Declining):     3.5 → 3.4 → 3.3 → 3.0  (DROPOUT RISK ⚠️)
Student B (Improving):     3.0 → 3.2 → 3.4 → 3.6  (LOW RISK ✓)
```

**Traditional ML Problem**: 
- Decision Trees, Random Forests, XGBoost see only the average (3.3)
- They miss the critical insight: **Student A is in free fall, Student B is recovering**

**AHFS-TA Solution**:
- Captures semester-by-semester progression
- Learns which semesters matter most (e.g., Semester 2-3 transitions are critical)
- Automatically selects features most predictive of dropout
- Combines 4 psychosocial features extracted from student engagement texts

---

## AHFS-TA Framework Overview

### High-Level Architecture

```
INPUT: Student Data (34 raw features + engagement texts)
  ├─ Academic: GPA, attendance, units passed, course failures
  ├─ Financial: Tuition status, scholarship, financial aid
  ├─ Demographic: Age, gender, parent education, distance to campus
  └─ Temporal: 4 semesters of data

    ↓↓↓ STAGE 1: LLM Feature Extraction ↓↓↓
    Enrich with 4 psychosocial features via DistilBERT:
    - Engagement Level (0-1)
    - Sentiment Score (0-1)
    - Topic Consistency (0-1)
    - Academic Motivation (0-1)
    → TOTAL: 38 features

    ↓↓↓ STAGE 2: Temporal Attention Network ↓↓↓
    Process 4 semesters through:
    - BiGRU (Forward & Backward): Captures temporal dependencies
    - 4-Head Attention: Identifies critical time periods
    → OUTPUT: 128-D temporal context vector
    
    ↓↓↓ STAGE 3: Adaptive Feature Selection (Epoch 10) ↓↓↓
    Three-stream ranking:
    - SHAP Importance (50% weight)
    - Correlation-based Importance (30% weight)
    - Temporal Variance (20% weight)
    Select TOP 28 features from 38
    → REDUCTION: 38 → 28 features (26.3% reduction)

    ↓↓↓ STAGE 4: Classification Head ↓↓↓
    Combine temporal context + selected features:
    - [128-D temporal] + [28 features] = 156-D vector
    - Dense Layer 1: 256 neurons
    - Dense Layer 2: 64 neurons
    - Output Layer: 3 classes (Dropout, Enrolled, Graduate)

OUTPUT: Probability scores for each class
  [0.89, 0.08, 0.03] → PREDICTED CLASS: Dropout
```

---

## Component 1: LLM-Based Semantic Feature Extraction

### Purpose
Extract psychosocial features from student engagement texts (emails, forum posts, comments) using DistilBERT, a lightweight BERT model.

### How It Works

#### Step 1: Text Collection
For each student, collect their engagement texts from a semester:
```
Student A - Semester 1 Engagement:
"I'm really excited about this course. The topics are fascinating and I'm 
keeping up with all the assignments. Looking forward to the projects!"

Student A - Semester 4 Engagement:
"I'm struggling to keep up. The workload feels overwhelming and I haven't 
started the last assignment. Not sure if I can handle this course."
```

#### Step 2: DistilBERT Embedding
Pass text through DistilBERT to get 768-dimensional embeddings:

```
Text: "I'm excited about this course..."
  ↓
DistilBERT Encoder
  ↓
Embedding: [0.23, -0.45, 0.78, ..., 0.12]  (768-D vector)
  ↓
Average pooling across all texts in semester
  ↓
Semester Embedding: [0.21, -0.42, 0.80, ..., 0.14]  (768-D)
```

#### Step 3: Extract 4 Psychosocial Features

From the 768-D embedding, extract 4 features using supervised fine-tuning:

```
Feature 1: Engagement Level
  Formula: Engagement = sigmoid(Linear_projection(embedding))
  Range: [0, 1]
  Interpretation: How actively is the student participating?
  Example:
    - Semester 1: 0.95 (very engaged)
    - Semester 4: 0.42 (disengaged)

Feature 2: Sentiment Score
  Formula: Sentiment = tanh(Linear_projection(embedding))
  Range: [-1, 1]
  Interpretation: Is the student positive or negative about coursework?
  Example:
    - Semester 1: 0.87 (positive, optimistic)
    - Semester 4: -0.52 (negative, struggling)

Feature 3: Topic Consistency
  Formula: Consistency = cosine_similarity(embedding, course_topics_embedding)
  Range: [0, 1]
  Interpretation: How on-topic is the student's communication?
  Example:
    - Semester 1: 0.89 (discusses relevant course concepts)
    - Semester 4: 0.45 (scattered, unfocused)

Feature 4: Academic Motivation
  Formula: Motivation = softmax(Linear_projection(embedding))[academic_class]
  Range: [0, 1]
  Interpretation: How motivated does the student sound about academics?
  Example:
    - Semester 1: 0.91 (intrinsically motivated)
    - Semester 4: 0.28 (low motivation)
```

### Concrete Example: Student A Evolution

```
SEMESTER 1:
  Raw features: GPA=3.8, Attendance=95%, Units_Passed=4
  LLM_Engagement=0.95, LLM_Sentiment=0.87, LLM_TopicConsistency=0.89, LLM_Motivation=0.91

SEMESTER 2:
  Raw features: GPA=3.6, Attendance=90%, Units_Passed=4
  LLM_Engagement=0.92, LLM_Sentiment=0.82, LLM_TopicConsistency=0.86, LLM_Motivation=0.88

SEMESTER 3:
  Raw features: GPA=2.9, Attendance=75%, Units_Passed=3
  LLM_Engagement=0.75, LLM_Sentiment=0.61, LLM_TopicConsistency=0.72, LLM_Motivation=0.65

SEMESTER 4:
  Raw features: GPA=1.5, Attendance=45%, Units_Passed=1
  LLM_Engagement=0.42, LLM_Sentiment=-0.52, LLM_TopicConsistency=0.45, LLM_Motivation=0.28
```

**Total Features Now: 38** (34 raw + 4 LLM-derived)

---

## Component 2: Temporal Attention Network

### Purpose
Capture how student behavior evolves across semesters and identify which time periods are most predictive of dropout.

### Why Temporal Modeling Matters

Without temporal modeling, a student who starts strong (S1: GPA 3.8) and crashes (S4: GPA 1.5) looks similar to a struggling-then-recovering student. **Temporal attention learns the difference.**

### Part A: BiGRU (Bidirectional Gated Recurrent Unit)

BiGRU processes the temporal sequence in TWO directions:

#### Direction 1: Forward (→) Past to Present
Process: Semester 1 → Semester 2 → Semester 3 → Semester 4

**Key equations for each semester t:**

```
Update Gate z_t = σ(W_z [h_{t-1}, x_t] + b_z)
  Purpose: How much to update hidden state (0=keep old, 1=use new)
  
Reset Gate r_t = σ(W_r [h_{t-1}, x_t] + b_r)
  Purpose: How much of previous state to forget
  
Candidate h̃_t = tanh(W_h [r_t ⊙ h_{t-1}, x_t] + b_h)
  Purpose: New information extracted from current semester
  
Final h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t
  Purpose: Blend old (retain) with new (learn)
```

**Numerical Example - Forward Direction:**

```
SEMESTER 1:
  Input x_1 = [3.8, 95, 4, 0.95, 0.87, 0.89, 0.91, ...]  (38-D)
  Previous hidden h_0 = [0, 0, 0, ..., 0]  (128-D, initialized)
  
  Update gate z_1 = σ(W_z·[h_0, x_1] + b_z)
                 = [0.54, 0.58, 0.57, ..., 0.53]  (128-D, values in [0,1])
  
  Hidden state h_1 = [0.13, -0.30, 0.23, ..., -0.10]  (128-D)
  → Captures: "Student is doing well"

SEMESTER 2:
  Input x_2 = [3.6, 90, 4, 0.92, 0.82, 0.86, 0.88, ...]  (38-D)
  Previous hidden h_1 = [0.13, -0.30, 0.23, ..., -0.10]  (128-D, NOT zero!)
  
  Update gate z_2 = σ(W_z·[h_1, x_2] + b_z)
                 = [0.61, 0.53, 0.59, ..., 0.50]  (128-D)
  
  Hidden state h_2 = [0.36, -0.14, 0.36, ..., 0.21]  (128-D)
  → Captures: "Student is still doing well, slight decline"

SEMESTER 3:
  Input x_3 = [2.9, 75, 3, 0.75, 0.61, 0.72, 0.65, ...]  (38-D)
  
  Hidden state h_3 = [0.52, 0.29, 0.46, ..., 0.38]  (128-D)
  → Captures: "Student is struggling"

SEMESTER 4:
  Input x_4 = [1.5, 45, 1, 0.42, -0.52, 0.45, 0.28, ...]  (38-D)
  
  Hidden state h_4_forward = [0.69, 0.41, 0.62, ..., 0.50]  (128-D)
  → Captures: "Student is in crisis"
```

#### Direction 2: Backward (←) Present to Past
Process: Semester 4 → Semester 3 → Semester 2 → Semester 1

```
h_4_backward = [0.75, 0.52, 0.63, ..., 0.59]  (128-D)
h_3_backward = [0.61, 0.40, 0.51, ..., 0.47]  (128-D)
h_2_backward = [0.49, 0.28, 0.39, ..., 0.35]  (128-D)
h_1_backward = [0.46, 0.23, 0.36, ..., 0.31]  (128-D)
```

**Why bidirectional?**
- Forward: "How does the past explain the present?"
- Backward: "What does the future reveal about the present?"
- Together: Bidirectional understanding of temporal patterns

#### Step 3: Concatenate Directions

```
h_t^bi = [h_t_forward; h_t_backward]

h_1^bi = [0.13, -0.30, 0.23, ..., -0.10 | 0.46, 0.23, 0.36, ..., 0.31]
       = 256-D vector (128 + 128)

h_2^bi = [0.36, -0.14, 0.36, ..., 0.21 | 0.49, 0.28, 0.39, ..., 0.35]
       = 256-D vector

h_3^bi = [0.52, 0.29, 0.46, ..., 0.38 | 0.61, 0.40, 0.51, ..., 0.47]
       = 256-D vector

h_4^bi = [0.69, 0.41, 0.62, ..., 0.50 | 0.75, 0.52, 0.63, ..., 0.59]
       = 256-D vector
```

**Output from BiGRU:** 4 vectors of 256-D each (one per semester)

---

### Part B: Multi-Head Attention

**Purpose:** Learn WHICH SEMESTERS are most important for predicting dropout.

#### Mechanism: Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

**What do Q, K, V mean?**
- Q (Query): "What am I looking for?" (from current semester)
- K (Key): "What information do I have?" (from all semesters)
- V (Value): "What is the actual information?" (from all semesters)

#### 4 Parallel Attention Heads

Each head learns different temporal patterns:

```
Head 1: Focuses on early-semester patterns (S1-S2 transitions)
Head 2: Focuses on mid-semester patterns (S2-S3 transitions)
Head 3: Focuses on recent patterns (S3-S4 behavior)
Head 4: Focuses on anomalies and sudden changes
```

#### Concrete Attention Calculation Example:

```
INPUT: 4 BiGRU outputs (256-D each)
  h_1^bi, h_2^bi, h_3^bi, h_4^bi

HEAD 1 PROCESSING:
  1. Project inputs: Q_1 = h_bi · W^Q (256×64), K_1 = h_bi · W^K (256×64), V_1 = h_bi · W^V (256×64)
  2. Compute scores: QK^T (4×4 matrix of attention scores)
  3. Scale: divide by √64 = 8
  4. Softmax: convert to attention weights (sum=1 per row)
  5. Apply: multiply weights by values

EXAMPLE ATTENTION WEIGHTS (for declining student A):

Query from Semester 1:
  Attention to [S1, S2, S3, S4] = [0.25, 0.23, 0.26, 0.26]
  → Fairly balanced (early semesters don't show clear dropout pattern)

Query from Semester 4:
  Attention to [S1, S2, S3, S4] = [0.15, 0.18, 0.32, 0.35]
  → Heavily weights recent semesters (where crisis is visible)

Final output for each head: 4×64 matrix (weighted combinations)
```

#### Step: Concatenate All Heads

```
4 heads × 64-D each = [64 + 64 + 64 + 64] = 256-D

Then project to 128-D:
  [256-D vector] × W^O (256×128 projection) = 128-D

FINAL TEMPORAL CONTEXT VECTOR: [0.78, 0.61, -0.23, 0.57, 0.45, ..., 0.52]
This 128-D vector encodes the student's entire 4-semester journey!
```

**What do these 128 dimensions represent?**
```
Dimensions 1-32:   Patterns learned by Head 1 (early trends)
Dimensions 33-64:  Patterns learned by Head 2 (mid transitions)
Dimensions 65-96:  Patterns learned by Head 3 (recent behavior)
Dimensions 97-128: Patterns learned by Head 4 (anomalies)
```

Each dimension is a **learned abstract feature** that captures temporal patterns.

---

## Component 3: Adaptive Hierarchical Feature Selection

### Purpose
Automatically identify and keep only the 28 most predictive features from 38, reducing noise and improving interpretability.

### Why Adaptive Selection?

**Problem with pre-training feature selection:**
- Can't account for learned feature importance from the model itself
- Different features may matter at different training stages

**AHFS Solution:**
- Performs feature selection DURING training (at epoch 10)
- Uses three independent ranking methods for consensus
- Weights them by reliability

### Three-Stream Importance Ranking

#### Stream 1: SHAP Importance (50% weight)

**SHAP (SHapley Additive exPlanations)** measures each feature's contribution using game theory.

```
For each feature i:
  SHAP_importance(i) = Shapley value of feature i
  
  Higher SHAP = Feature has larger impact on model predictions
```

**Concrete Example:**

```
Feature: LLM_Engagement

To compute SHAP importance:
  1. Train Random Forest on [all features]
  2. Measure prediction change when LLM_Engagement is removed
  3. SHAP_importance(LLM_Engagement) = average marginal contribution
  
For Student A:
  Prediction with all features: 0.89 (89% dropout)
  Prediction without LLM_Engagement: 0.71 (71% dropout)
  SHAP_importance = 0.89 - 0.71 = 0.18 (18% impact)
  
Normalized to [0,1]: SHAP_norm(LLM_Engagement) = 1.0 (highest impact)
```

#### Stream 2: Correlation-Based Importance (30% weight)

**Pearson Correlation** measures linear relationship with target outcome.

```
For each feature i:
  Corr_importance(i) = |correlation(feature_i, dropout_target)|
  
  Range: [0, 1]
  Higher = Stronger linear relationship with dropout
```

**Concrete Example:**

```
Feature: Attendance_Rate

Compute correlation with dropout labels:
  - Students who dropped out: average attendance = 45%
  - Students who didn't dropout: average attendance = 78%
  
Pearson correlation coefficient: -0.72 (negative, as expected)
Absolute value: 0.72
Normalized: Corr_norm(Attendance) = 0.72
```

#### Stream 3: Temporal Variance Importance (20% weight)

**Variance** measures how much a feature changes across semesters.

```
For each feature i:
  Temp_importance(i) = Variance({x_i,1, x_i,2, x_i,3, x_i,4})
  
  Higher variance = Feature is dynamic, captures changing behavior
```

**Concrete Example:**

```
Feature 1: Attendance Rate
  Values across semesters: [95, 90, 75, 45]
  Mean: 76.25
  Variance = ((95-76.25)² + (90-76.25)² + (75-76.25)² + (45-76.25)²) / 4
           = (351.56 + 189.06 + 1.56 + 976.56) / 4
           = 379.69 (HIGH variance → important dynamic signal)

Feature 2: GPA
  Values: [3.8, 3.6, 2.9, 1.5]
  Variance = 0.81 (moderate)

Feature 3: Scholarship Status
  Values: [1, 1, 0, 0]
  Variance = 0.25 (low variance → stable feature)
```

### Fusion: Meta-Importance Score

**Combine all three streams:**

$$\text{Meta-Importance}(i) = 0.5 \cdot \text{SHAP}_{norm}(i) + 0.3 \cdot \text{Corr}_{norm}(i) + 0.2 \cdot \text{Temp}_{norm}(i)$$

**Example Calculation:**

```
Feature: LLM_Engagement
  SHAP_norm = 1.0   (most important for model)
  Corr_norm = 1.0   (strongest correlation with dropout)
  Temp_norm = 0.95  (high variance across semesters)
  
  Meta-Importance = 0.5(1.0) + 0.3(1.0) + 0.2(0.95)
                  = 0.50 + 0.30 + 0.19
                  = 0.99 (RANK #1)

Feature: Attendance_Rate
  SHAP_norm = 0.92
  Corr_norm = 0.92
  Temp_norm = 0.95
  
  Meta-Importance = 0.5(0.92) + 0.3(0.92) + 0.2(0.95)
                  = 0.46 + 0.276 + 0.19
                  = 0.926 (RANK #2)

Feature: Housing_Type
  SHAP_norm = 0.15
  Corr_norm = 0.18
  Temp_norm = 0.05
  
  Meta-Importance = 0.5(0.15) + 0.3(0.18) + 0.2(0.05)
                  = 0.075 + 0.054 + 0.01
                  = 0.139 (RANK #28 - borderline for selection)
```

### Feature Selection Result

**All 38 features ranked and sorted:**

```
RANK | FEATURE                  | META-SCORE | STATUS
-----|--------------------------|------------|--------
1    | LLM_Engagement          | 0.99      | ✓ SELECTED
2    | S2_Approved_Units       | 0.92      | ✓ SELECTED
3    | LLM_Sentiment           | 0.87      | ✓ SELECTED
4    | S2_Grade                | 0.86      | ✓ SELECTED
5    | S1_Grade                | 0.85      | ✓ SELECTED
...
28   | Extracurricular_Inv     | 0.14      | ✓ SELECTED (borderline)
29   | Housing_Type            | 0.13      | ✗ NOT SELECTED
30   | Employment_Status       | 0.11      | ✗ NOT SELECTED
31   | Accommodation_Type      | 0.09      | ✗ NOT SELECTED
32   | Marital_Status          | 0.08      | ✗ NOT SELECTED
...
38   | Random_Noise_Feature    | 0.01      | ✗ NOT SELECTED

REDUCTION: 38 features → 28 features (removed 26.3%)
```

**Why this works:**
- SHAP ensures selected features impact model predictions
- Correlation ensures statistical relevance
- Temporal variance ensures dynamic signals
- Consensus avoids over-reliance on single method

---

## Component 4: Classification Head

### Purpose
Combine temporal context with selected features to make final dropout/enrolled/graduate predictions.

### Architecture

```
INPUT: [128-D temporal context] + [28 selected features] = 156-D vector

        ↓
    Dense Layer 1
        Input: 156-D
        Weights: 156 × 256 matrix
        Activation: ReLU
        Dropout: 0.3 (drop 30% of neurons randomly)
        Output: 256-D
        
        ↓
    Dense Layer 2
        Input: 256-D (from Layer 1)
        Weights: 256 × 64 matrix
        Activation: ReLU
        Dropout: 0.3
        Output: 64-D
        
        ↓
    Output Layer
        Input: 64-D (from Layer 2)
        Weights: 64 × 3 matrix
        Activation: Softmax
        Output: 3-D probability vector
        
OUTPUT: [P(Dropout), P(Enrolled), P(Graduate)]
        Example: [0.89, 0.08, 0.03]
        Predicted Class: DROPOUT (highest probability)
```

### Example Forward Pass

**Student A Data (Semester 4):**

```
TEMPORAL CONTEXT (128-D):
  From Component 2: [0.78, 0.61, -0.23, 0.57, 0.45, 0.68, 0.46, 0.52, -0.18, 0.61, ...]

SELECTED FEATURES (28-D):
  [LLM_Engagement=0.42, S2_Units=3, LLM_Sentiment=-0.52, S2_Grade=3.2, S1_Grade=3.8, ...]

CONCATENATION (156-D):
  [0.78, 0.61, -0.23, 0.57, 0.45, 0.68, ... (128 values) ... | 0.42, 3, -0.52, 3.2, 3.8, ... (28 values) ...]

LAYER 1 (156 → 256):
  z_1 = ReLU(W_1 · [156-D] + b_1)
  
  Before ReLU: [-0.45, 0.89, -0.12, 0.67, ...]  (256 values, some negative)
  After ReLU: [0, 0.89, 0, 0.67, ...]  (negative values become 0)
  After Dropout(0.3): [0, 0.89, 0, 0.67, ...]  (30% randomly zeroed)
  
  Output: 256-D vector: [0.0, 0.89, 0.0, 0.67, 0.34, ..., 0.21]

LAYER 2 (256 → 64):
  z_2 = ReLU(W_2 · [256-D] + b_2)
  
  Output: 64-D vector: [0.45, 0.12, 0.78, 0.34, 0.67, ..., 0.23]

LAYER 3 - OUTPUT (64 → 3):
  logits = W_3 · [64-D] + b_3
  logits = [2.15, -0.87, -1.92]  (raw scores)
  
  Apply Softmax:
    e^2.15 = 8.59,  e^-0.87 = 0.42,  e^-1.92 = 0.15
    Sum = 8.59 + 0.42 + 0.15 = 9.16
    
  Probabilities:
    P(Dropout)  = 8.59 / 9.16 = 0.94
    P(Enrolled) = 0.42 / 9.16 = 0.05
    P(Graduate) = 0.15 / 9.16 = 0.01
    
FINAL OUTPUT: [0.94, 0.05, 0.01]
PREDICTION: DROPOUT (94% confidence)
```

### Loss Function: Weighted Cross-Entropy

To handle class imbalance (Graduate: 50%, Dropout: 32%, Enrolled: 18%):

```
Loss = -Σ(over all students and classes) w_c · y_c · log(ŷ_c)

Where:
  w_c = N / (3 · N_c)  (inverse frequency weighting)
  
For our dataset (N=4,424 students):
  w_Dropout = 4424 / (3 × 1421) = 1.04  (slightly more penalty)
  w_Enrolled = 4424 / (3 × 794) = 1.86  (highest penalty - rarest class)
  w_Graduate = 4424 / (3 × 2209) = 0.67 (lowest penalty - most common)
```

**Effect:** Misclassifying an Enrolled student is penalized 1.86× more than Graduate.

---

## Algorithm 2: Complete Training Procedure

### Two-Phase Training Strategy

The entire training is divided into **two distinct phases**, with adaptive feature selection occurring at the boundary.

### PHASE 1: Initial Training (Epochs 1-10)

**All 38 features used** while building importance signals.

```
FOR epoch = 1 to 10:
  FOR each batch (x, y) in training_data:
  
    1. FORWARD PASS:
       - Input: batch of 38 features (4 semesters)
       - Component 1 (LLM): Already applied, features included
       - Component 2 (Temporal): BiGRU + Attention → 128-D
       - Component 3: Not active yet (all 38 features used)
       - Component 4: Classify from [128-D temporal + 38 features] = 166-D
       
       ŷ_batch = AHFS-TA(x; θ)
    
    2. LOSS CALCULATION:
       L = WeightedCrossEntropy(y_batch, ŷ_batch)
       
       Example loss: 0.45 (good fit)
    
    3. BACKWARD PASS:
       ∇_θ L = compute gradients
       θ = θ - η · AdamW(∇_θ L)
       
       Where:
         η = learning rate (from cosine schedule)
         AdamW = adaptive learning rate with weight decay
         θ = all model parameters
```

**During Phase 1:**
- Model learns patterns on ALL 38 features
- Accumulates SHAP importance scores
- Accumulates correlation statistics
- Accumulates temporal variance information

### ADAPTIVE FEATURE SELECTION (Epoch 10 → Epoch 11)

**Critical checkpoint where features are selected!**

```
At END of Epoch 10:

1. COMPUTE IMPORTANCE SCORES:
   
   SHAP Stream:
     For each of 38 features:
       importance_shap(i) = Shapley value from trained Random Forest
   
   Correlation Stream:
     For each of 38 features:
       importance_corr(i) = |Pearson correlation with dropout target|
   
   Temporal Variance Stream:
     For each of 38 features:
       importance_temp(i) = Variance across 4 semesters

2. NORMALIZE to [0, 1]:
   importance_shap_norm = normalize(importance_shap)
   importance_corr_norm = normalize(importance_corr)
   importance_temp_norm = normalize(importance_temp)

3. FUSE WITH WEIGHTS:
   For i = 1 to 38:
     Meta_importance(i) = 0.5 · importance_shap_norm(i)
                        + 0.3 · importance_corr_norm(i)
                        + 0.2 · importance_temp_norm(i)

4. SELECT TOP 28 FEATURES:
   Sort all 38 features by Meta_importance (descending)
   Keep top K = 28 features
   Discard bottom 10 features

5. REINITIALIZE MODEL:
   - Input layer: change from 38 to 28 features
   - Optimizer: reset learning rate
   - Learning rate scheduler: reset
   - Model ready for Phase 2
```

### PHASE 2: Fine-Tuning with Selected Features (Epochs 11-50)

**Only 28 features used** for focused learning.

```
FOR epoch = 11 to 50:
  FOR each batch (x, y) in training_data:
  
    1. FEATURE SELECTION:
       x_selected = x[:, top_28_features]  (extract only selected features)
    
    2. FORWARD PASS:
       ŷ_batch = AHFS-TA(x_selected; θ)
       
       Now input dimension reduced: [128-D temporal + 28 features] = 156-D
    
    3. LOSS & UPDATE:
       L = WeightedCrossEntropy(y_batch, ŷ_batch)
       θ = θ - η_t · AdamW(∇_θ L)
  
  2. VALIDATION CHECK:
     accuracy_val = Evaluate(θ on validation_set)
     
     IF accuracy_val > best_accuracy:
       best_accuracy = accuracy_val
       Save checkpoint(θ)
       patience_counter = 0
     ELSE:
       patience_counter = patience_counter + 1
     
     IF patience_counter ≥ 10:
       BREAK (Early stopping)

RETURN Best saved model (highest validation accuracy)
```

### Learning Rate Schedule

**Cosine annealing:** Smooth decay from initial to final learning rate

$$\eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})(1 + \cos(t \cdot \pi/T))$$

```
η_max = 0.001 (initial)
η_min = 0.0001 (final)
T = 50 (total epochs)

Epoch 1:  η = 0.001   (aggressive learning)
Epoch 10: η = 0.00076 (moderate)
Epoch 25: η = 0.00055 (fine-tuning)
Epoch 40: η = 0.00025 (convergence)
Epoch 50: η = 0.0001  (minimal updates)
```

### Early Stopping

```
IF validation_accuracy doesn't improve for 10 consecutive epochs:
  STOP training immediately
  
Why: Prevents overfitting and saves computation
Example:
  Epoch 15: val_acc = 0.89 ✓ (best)
  Epoch 16: val_acc = 0.888 (patience = 1)
  Epoch 17: val_acc = 0.887 (patience = 2)
  ...
  Epoch 25: val_acc = 0.885 (patience = 10) → STOP!
```

---

## End-to-End Example with Real Data

Let me walk through the complete AHFS-TA pipeline with a concrete student example.

### Student Profile: Sarah (At-Risk Student)

```
BASIC INFO:
  Name: Sarah
  Program: Computer Science
  Student ID: 12345
  Classes: 3 (Dropout, Enrolled, Graduate)
  Expected: Enrolled, Actual: Dropout
```

### Raw Data (34 Features across 4 Semesters)

#### Academic Features:
```
SEMESTER 1 (Good Performance):
  GPA: 3.8 / 4.0
  Attendance_Rate: 95%
  Units_Enrolled: 4
  Units_Passed: 4
  Units_Failed: 0
  Average_Grade: 3.8
  
SEMESTER 2 (Slight Decline):
  GPA: 3.6 / 4.0
  Attendance_Rate: 90%
  Units_Enrolled: 4
  Units_Passed: 4
  Units_Failed: 0
  Average_Grade: 3.6
  
SEMESTER 3 (Significant Decline):
  GPA: 2.9 / 4.0
  Attendance_Rate: 75%
  Units_Enrolled: 4
  Units_Passed: 3
  Units_Failed: 1
  Average_Grade: 2.9
  
SEMESTER 4 (Critical Crisis):
  GPA: 1.5 / 4.0
  Attendance_Rate: 45%
  Units_Enrolled: 4
  Units_Passed: 1
  Units_Failed: 3
  Average_Grade: 1.5
```

#### Financial Features:
```
Tuition_Fee_Status: 1 (Paid)
Scholarship_Holder: 1 (Yes) → 0 (Lost in S3)
Financial_Aid_Amount: 5000
Family_Income_Level: 3 (Middle)
```

#### Demographic Features:
```
Age: 21
Gender: 1 (Female)
Parent_Education: 2 (Secondary)
Distance_to_Campus: 15 km
Previous_GPA: 3.5
First_Generation: 0 (No)
```

#### Engagement Texts (for LLM extraction):

```
SEMESTER 1:
"I'm so excited about starting this CS program! The courses are fascinating 
and I'm doing well in all of them. I love the algorithms class especially. 
Really looking forward to learning more."

→ LLM_Engagement: 0.95 (very engaged)
→ LLM_Sentiment: 0.87 (positive, optimistic)
→ LLM_TopicConsistency: 0.89 (focused on course topics)
→ LLM_Motivation: 0.91 (intrinsic motivation)

SEMESTER 2:
"Still enjoying the course but starting to feel the workload. Some assignments 
are getting harder but I'm keeping up. Hope it gets easier."

→ LLM_Engagement: 0.92
→ LLM_Sentiment: 0.82
→ LLM_TopicConsistency: 0.86
→ LLM_Motivation: 0.88

SEMESTER 3:
"Struggling a lot now. The material is getting really hard and I feel lost 
in most classes. Not sure what's happening to me."

→ LLM_Engagement: 0.75
→ LLM_Sentiment: 0.61
→ LLM_TopicConsistency: 0.72
→ LLM_Motivation: 0.65

SEMESTER 4:
"I can't handle this anymore. I've already failed 3 courses and I don't see 
how I can fix this. Everything feels overwhelming and pointless now."

→ LLM_Engagement: 0.42 (disengaged)
→ LLM_Sentiment: -0.52 (negative, hopeless)
→ LLM_TopicConsistency: 0.45 (scattered, unfocused)
→ LLM_Motivation: 0.28 (lost motivation)
```

### STAGE 1: LLM Feature Extraction Complete ✓

```
Final feature set: 34 + 4 = 38 features
Ready for temporal processing
```

### STAGE 2: Temporal Attention Network Processing

#### BiGRU Forward Direction:

```
h_0_forward = [0, 0, 0, ..., 0]  (128-D, initialized)

SEMESTER 1:
  x_1 = [3.8, 95, 4, 4, 0, 3.8, 1, 0, 5000, 3, 21, 1, 2, 15, 3.5, 0,
         0.95, 0.87, 0.89, 0.91, ...]  (38-D)
  
  z_1 = σ(W_z·[h_0, x_1] + b_z) = [0.54, 0.58, 0.57, ..., 0.53]  (128-D)
  r_1 = σ(W_r·[h_0, x_1] + b_r) = [0.61, 0.49, 0.52, ..., 0.45]  (128-D)
  h̃_1 = tanh(W_h·[r_1⊙h_0, x_1] + b_h) = [0.24, -0.52, 0.41, ..., -0.19]
  
  h_1_forward = (1-z_1)⊙h_0 + z_1⊙h̃_1
              = [0.13, -0.30, 0.23, ..., -0.10]  (128-D)

SEMESTER 2:
  x_2 = [3.6, 90, 4, 4, 0, 3.6, 1, 0, 5000, 3, 21, 1, 2, 15, 3.5, 0,
         0.92, 0.82, 0.86, 0.88, ...]  (38-D)
  
  h_2_forward = [0.36, -0.14, 0.36, ..., 0.21]  (128-D)

SEMESTER 3:
  x_3 = [2.9, 75, 4, 3, 1, 2.9, 1, 0, 5000, 3, 21, 1, 2, 15, 3.5, 0,
         0.75, 0.61, 0.72, 0.65, ...]  (38-D)
  
  h_3_forward = [0.52, 0.29, 0.46, ..., 0.38]  (128-D)

SEMESTER 4:
  x_4 = [1.5, 45, 4, 1, 3, 1.5, 0, 0, 5000, 3, 21, 1, 2, 15, 3.5, 0,
         0.42, -0.52, 0.45, 0.28, ...]  (38-D)
  
  h_4_forward = [0.69, 0.41, 0.62, ..., 0.50]  (128-D)
```

#### BiGRU Backward Direction:

```
h_4_backward = [0.75, 0.52, 0.63, ..., 0.59]  (128-D)
h_3_backward = [0.61, 0.40, 0.51, ..., 0.47]  (128-D)
h_2_backward = [0.49, 0.28, 0.39, ..., 0.35]  (128-D)
h_1_backward = [0.46, 0.23, 0.36, ..., 0.31]  (128-D)
```

#### Bidirectional Concatenation:

```
h_4^bi = [0.69, 0.41, 0.62, ..., 0.50 | 0.75, 0.52, 0.63, ..., 0.59]
       = 256-D vector (this is used for attention)
```

#### Multi-Head Attention (Focusing on Semester 4):

```
HEAD 1 (Early patterns):
  Computes attention from S1 context
  Learns that S1 was good, so probably not due to lack of ability
  Attention weights to [S1, S2, S3, S4]: [0.25, 0.24, 0.25, 0.26]

HEAD 2 (Mid-semester transitions):
  Computes attention from S2-S3 context
  Learns about the critical transition point
  Attention weights to [S1, S2, S3, S4]: [0.18, 0.22, 0.32, 0.28]

HEAD 3 (Recent behavior):
  Computes attention from recent context
  HEAVILY focuses on S4 (the crisis)
  Attention weights to [S1, S2, S3, S4]: [0.10, 0.15, 0.30, 0.45]

HEAD 4 (Anomalies):
  Detects the sudden drop
  Largest attention on S4 (worst performance)
  Attention weights to [S1, S2, S3, S4]: [0.08, 0.12, 0.28, 0.52]
```

#### Attention Output & Projection:

```
Concatenate 4 heads: [64 + 64 + 64 + 64] = 256-D
Project: W^O (256×128) = 128-D

TEMPORAL_CONTEXT = [0.78, 0.61, -0.23, 0.57, 0.45, 0.68, 0.46, 0.52, 
                    -0.18, 0.61, 0.34, 0.49, 0.82, 0.35, 0.59, 0.27,
                    0.41, 0.71, -0.12, 0.63, ... (128 total values)]

This 128-D vector ENCODES:
  ✓ Early success (S1-S2)
  ✓ Transition crisis (S2→S3)
  ✓ Escalating decline (S3→S4)
  ✓ Anomalies (sudden drop in engagement)
  ✓ Trajectory: DOWNWARD SPIRAL
```

### STAGE 3: Adaptive Feature Selection (at Epoch 10)

#### Importance Calculation (simplified):

```
All 38 features ranked:

RANK | FEATURE              | SHAP | CORR | TEMP | META  | SELECTED
-----|----------------------|------|------|------|-------|----------
1    | LLM_Engagement      | 1.0  | 1.0  | 0.95 | 0.99  | ✓
2    | S2_Approved_Units   | 0.92 | 0.92 | 0.88 | 0.92  | ✓
3    | LLM_Sentiment       | 0.90 | 0.88 | 0.85 | 0.87  | ✓
4    | S2_Grade            | 0.88 | 0.90 | 0.82 | 0.86  | ✓
5    | S1_Grade            | 0.85 | 0.87 | 0.78 | 0.85  | ✓
6    | S1_Approved_Units   | 0.84 | 0.86 | 0.76 | 0.84  | ✓
7    | Tuition_Fee_Status  | 0.72 | 0.68 | 0.44 | 0.68  | ✓
8    | LLM_TopicConsistency| 0.78 | 0.68 | 0.48 | 0.68  | ✓
9    | Scholarship_Holder  | 0.65 | 0.72 | 0.68 | 0.68  | ✓
10   | S3_Grade            | 0.75 | 0.70 | 0.60 | 0.70  | ✓
... (11-28 all selected, space saving)
28   | Extracurricular_Inv | 0.20 | 0.18 | 0.15 | 0.19  | ✓
29   | Housing_Type        | 0.18 | 0.16 | 0.08 | 0.15  | ✗
30   | Employment_Status   | 0.16 | 0.14 | 0.06 | 0.12  | ✗
...
38   | Random_Feature      | 0.05 | 0.03 | 0.02 | 0.04  | ✗

REMOVED: Features 29-38 (10 features with lowest meta-importance)
REDUCTION: 38 → 28 features
```

### STAGE 4: Classification Head

#### Input Preparation:

```
TEMPORAL_CONTEXT (128-D): [0.78, 0.61, -0.23, 0.57, 0.45, 0.68, ..., 0.52]

SELECTED_FEATURES (28-D):
  [LLM_Engagement=0.42, S2_Approved_Units=3, LLM_Sentiment=-0.52, S2_Grade=3.2,
   S1_Grade=3.8, S1_Approved_Units=4, Tuition=1, LLM_TopicConsistency=0.45,
   Scholarship=0, S3_Grade=2.9, S3_Approved_Units=3, Age=21, S4_Grade=1.5,
   S4_Approved_Units=1, Attendance=45, Parent_Education=2, Financial_Aid=5000,
   Gender=1, S1_Attendance=95, S2_Attendance=90, Distance=15, Previous_GPA=3.5,
   S3_Attendance=75, Work_Status=1, S4_Attendance=45, Housing=0,
   Marital_Status=0, LLM_Motivation=0.28]

CONCATENATION: [128-D] + [28-D] = 156-D vector
```

#### Dense Layer 1 (156 → 256):

```
z_1 = ReLU(W_1 · [156-D] + b_1)

Before activation: [-0.45, 0.89, -0.12, 0.67, 0.34, ..., 0.21]
After ReLU: [0, 0.89, 0, 0.67, 0.34, ..., 0.21]  (negative → 0)
After Dropout(0.3): [0, 0.89, 0, 0.67, 0, ..., 0.21]  (30% randomly zeroed)

Output_L1: [0.0, 0.89, 0.0, 0.67, 0.0, 0.45, 0.23, 0.78, ..., 0.15]  (256-D)
```

#### Dense Layer 2 (256 → 64):

```
z_2 = ReLU(W_2 · [256-D] + b_2)

After ReLU & Dropout:
Output_L2: [0.45, 0.12, 0.78, 0.34, 0.67, ..., 0.23]  (64-D)
```

#### Output Layer (64 → 3):

```
logits = W_3 · [64-D] + b_3
       = [2.15, -0.87, -1.92]  (raw scores)

Softmax:
  e^2.15 = 8.59   (Dropout)
  e^-0.87 = 0.42  (Enrolled)
  e^-1.92 = 0.15  (Graduate)
  Sum = 9.16

PROBABILITIES:
  P(Dropout)  = 8.59 / 9.16 = 0.938 (93.8%)
  P(Enrolled) = 0.42 / 9.16 = 0.046 (4.6%)
  P(Graduate) = 0.15 / 9.16 = 0.016 (1.6%)

PREDICTION: DROPOUT (Confidence: 93.8%)
```

### Prediction Summary:

```
SARAH'S PREDICTION:
  ✓ Predicted: DROPOUT (93.8% confidence)
  ✓ Actual: DROPOUT
  ✓ CORRECT! ✓

Model successfully identifies Sarah's high dropout risk because:
  1. Temporal analysis shows: Consistent decline across 4 semesters
  2. Critical features reveal:
     - LLM engagement dropped from 0.95 → 0.42
     - GPA fell from 3.8 → 1.5
     - Attendance declined from 95% → 45%
  3. Attention mechanism weighted recent semesters (S3-S4) heavily
  4. Feature selection kept only most predictive signals
```

---

## Why AHFS-TA Works Better

### Comparison with Baseline Models

#### Traditional ML Models (Decision Tree, Random Forest, XGBoost):
```
❌ Treat data as static snapshot (ignore temporal evolution)
❌ Can't distinguish between declining and recovering students
❌ Use all features equally (noise from irrelevant features)
❌ No semantic understanding of engagement
❌ Limited to linear/tree-based relationships

EXAMPLE FAILURE:
  Sarah's cumulative GPA: 3.3
  Baseline predicts: "Likely enrolled" (based on 3.3 average)
  
  BUT Sarah's trajectory:
    3.8 → 3.6 → 2.9 → 1.5 (DOWNWARD SPIRAL)
  Actually: DROPOUT
  
  Baseline: WRONG! ❌
```

#### Traditional Neural Networks (Basic LSTM/RNN):
```
✓ Capture temporal patterns
❌ Still use all 38 features (noisy inputs)
❌ No interpretability (black box)
❌ No semantic feature enrichment
❌ Can't explain which features matter

ACCURACY: ~71% (from our paper)
```

#### AHFS-TA Advantages:

```
✓ TEMPORAL MODELING (BiGRU + Multi-Head Attention)
  - Learns trajectory, not just values
  - Identifies critical semesters (S2-S3 transitions)
  - Distinguishes declining from recovering students
  - Example: Sarah (declining) vs. Student B (recovering) are correctly separated

✓ ADAPTIVE FEATURE SELECTION
  - Removes 26.3% of noise (38 → 28 features)
  - Consensus-based (SHAP + Correlation + Temporal Variance)
  - Updated during training (adapts to learned patterns)
  - Example: Removes spurious correlations like "housing type"

✓ SEMANTIC ENRICHMENT (LLM Features)
  - Captures engagement from text (not just grades)
  - LLM_Engagement, LLM_Sentiment, LLM_TopicConsistency
  - More predictive than behavioral metrics alone
  - Example: Sarah's sentiment shift (-0.52 in S4) is strong dropout signal

✓ EXPLAINABILITY (SHAP Integration)
  - Every feature has interpretable importance score
  - Can explain to students/advisors: "Why did we predict dropout?"
  - Example: "LLM Engagement (rank #1) and Grade (rank 4) were key factors"

✓ BALANCED PERFORMANCE (Weighted Loss)
  - Handles class imbalance (Enrolled is rare)
  - Enrolled class has 1.86× penalty weight
  - Prevents bias toward majority class (Graduate)
```

### Performance Comparison:

```
MODEL              | ACCURACY | AUC-ROC | NOTES
-------------------|----------|---------|------------------
Baseline Models:
  Decision Tree     | 67.0%    | 0.62    | Too simple, poor temporal
  Naive Bayes       | 70.9%    | 0.68    | Independence assumption
  Random Forest     | 76.7%    | 0.72    | Best baseline, still misses patterns
  AdaBoost          | 74.2%    | 0.70    | Ensemble benefit limited
  XGBoost           | 75.9%    | 0.71    | Non-linear, but no temporal
  Neural Network    | 71.4%    | 0.69    | Basic RNN, limited explainability

AHFS-TA (This Work):
  AHFS-TA           | 91.32%   | 0.955   | SOTA! Temporal + Features + LLM + Explainability

IMPROVEMENT:
  Over best baseline (Random Forest): +14.62% accuracy, +23.5% AUC-ROC
  Over neural baseline: +19.92% accuracy, +26.5% AUC-ROC
```

### Key Innovations in AHFS-TA:

```
1. FIRST framework integrating ALL THREE dimensions:
   ✓ Sophisticated temporal modeling (BiGRU + Multi-Head Attention)
   ✓ Multimodal feature fusion (academic + financial + LLM)
   ✓ Unified explainability (SHAP + Correlation + Temporal Variance)

2. NOVEL adaptive feature selection approach:
   ✓ Consensus-based (3 streams prevent single-method bias)
   ✓ During-training (adapts to learned patterns, not pre-training)
   ✓ Weighted fusion (reliability-based weights)

3. LLM-ENHANCED features:
   ✓ DistilBERT for semantic feature extraction
   ✓ Captures engagement not visible in grades alone
   ✓ Psychosocial signals highly predictive of dropout

4. COMPLETE pipeline:
   ✓ From raw data to explainable predictions
   ✓ Production-ready implementation
   ✓ Actionable insights for educational administrators
```

---

## Mathematical Notation Summary

### Key Variables:

```
x_t ∈ ℝ^38         Input features for semester t
h_t ∈ ℝ^128        Hidden state (GRU forward)
h̄_t ∈ ℝ^128        Hidden state (GRU backward)
h_t^bi ∈ ℝ^256     Bidirectional concatenation
Q, K, V ∈ ℝ^(T×64) Query, Key, Value projections
α ∈ ℝ^(T×T)        Attention weights
c ∈ ℝ^128          Temporal context vector
z_t, r_t ∈ [0,1]^128 Update and reset gates
θ                  All model parameters
η_t                Learning rate at epoch t
L                  Loss function
ŷ                  Predicted probabilities
```

### Key Operations:

```
σ(x)       Sigmoid function: 1/(1+e^(-x))
tanh(x)    Hyperbolic tangent: (e^x - e^(-x))/(e^x + e^(-x))
⊙          Element-wise (Hadamard) multiplication
[·;·]      Vector concatenation
∘          Function composition
∇_θ        Gradient with respect to θ
∝          Proportional to
```

---

## Conclusion

AHFS-TA achieves **state-of-the-art student dropout prediction** by synthesizing:

1. **Temporal Attention** to capture behavior evolution
2. **Adaptive Feature Selection** to focus on predictive signals
3. **LLM-Enrichment** for semantic understanding
4. **Explainable AI** for actionable insights

The framework demonstrates that **combining temporal modeling, feature-level interpretability, and semantic enrichment** produces both better predictions AND better understanding for educational decision-makers.

---

**Document prepared for: [Supervisor Name]**  
**Date: January 22, 2026**  
**Student: [Your Name]**
