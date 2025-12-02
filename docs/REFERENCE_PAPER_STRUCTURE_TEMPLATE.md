# Reference Paper Structure Template
## Extracted from: "Analyzing students' academic performance using educational data mining"

**Source:** Computers and Education: Artificial Intelligence, Vol 7 (2024)  
**Citation Style:** APA (author-year)  
**Document Type:** Academic Journal Article  

---

## 1. DOCUMENT STRUCTURE (Section Hierarchy)

### Complete Section Order:

1. **Abstract** (Keywords-based)
2. **1. Introduction**
   - 1.1. Research objectives
3. **2. Literature review**
4. **3. Data mining techniques** (Method descriptions)
   - 3.1. Decision tree
   - 3.2. k-Nearest Neighbor (k-NN)
   - 3.3. Naïve Bayes
   - 3.4. Neural networks
   - 3.5. Random forest
5. **4. Dataset and experimental methodology**
   - 4.1. Dataset
   - 4.2. Experimental methodology
6. **5. Results analysis**
   - 5.1. Objective 1 - [specific analysis]
   - 5.2. Objective 2 - [specific analysis]
   - 5.3. Objective 3 - [specific analysis]
   - 5.4. Objective 4 - [specific analysis]
7. **6. Limitations and future work**
8. **7. Theoretical and pedagogical implications**
9. **8. Conclusion**
10. **CRediT authorship contribution statement**
11. **Declaration of competing interest**
12. **Data availability**
13. **References**

---

## 2. METHODOLOGY SECTION STRUCTURE (Section 4)

### Main Title:
**4. Dataset and experimental methodology**

### Subsection Organization:

#### **4.1. Dataset**
- **Purpose:** Describe data sources, characteristics, and attributes
- **Content structure:**
  1. Dataset overview and source description
  2. Table presenting dataset characteristics (e.g., Table 1: Dataset characteristics)
  3. Detailed attribute descriptions (e.g., Table 2: Attributes in dataset)
  4. Class distribution statistics (e.g., Table 3 & 4: Statistics of class-wise student distribution)
  5. Data ranges and categorization criteria
  6. Implementation tools mentioned (e.g., "RapidMiner software and manually coded at Google Colaboratory")

#### **4.2. Experimental methodology**
- **Purpose:** Explain the experimental design, approach, and workflow
- **Content structure:**
  1. Context and motivation for the approach
  2. Research questions being addressed
  3. Analysis cases/scenarios description
  4. Classification algorithms and criteria used
  5. Evaluation techniques (e.g., "5-fold cross-validation technique")
  6. Preliminary analysis steps
  7. **Workflow diagram** (e.g., Fig. 6: Workflow of proposed methodology)
  8. **Mathematical formulations** for each approach:
     - Numbered equations (1), (2), (3), etc.
     - Variable definitions immediately after equations
  9. Detailed explanation of each research objective's methodology
  10. Performance metrics definitions (equations 7-11 for accuracy, precision, recall, F1-score, Kappa)

### Key Pattern:
- **Methodology is organized by research objectives** (Objective 1, 2, 3, 4)
- Each objective has its own methodological approach explained
- Workflow proceeds from data → processing → analysis → evaluation

---

## 3. MATHEMATICAL NOTATION STYLE

### Equation Formatting:
- **Numbered equations:** Yes, consecutively numbered (1), (2), (3)... up to (11)
- **Placement:** Centered, on separate lines
- **Numbering position:** Right-aligned in parentheses

### Example Format:
```
𝐻𝑆𝐶_GPA = (𝐽𝑆𝐶_GPA × 0.25) + (𝑆𝑆𝐶_GPA × 0.75)     (2)
```

### Variable Definitions:
- **Placed immediately after equation**
- **Format:** "Here, [variable] is [definition]"
- Example:
  ```
  Here, 𝐽𝑆𝐶_GPA is obtained GPA in JSC examination
        𝑆𝑆𝐶_GPA is obtained GPA in SSC examination
  ```

### Mathematical Symbols Used:
- Summation: ∑
- Multiplication: ×
- Division: standard fraction notation
- Subscripts and superscripts for indexing

### Notation Conventions:
- **Italics** for variables (GPA, n, i)
- **Subscripts** for indices (Sub_i, n₁)
- **Greek letters** for statistical measures (not prominent in this paper)

---

## 4. TABLES AND FIGURES

### Table Format:

#### Table Structure:
- **Table number and caption ABOVE the table**
- **Caption format:** "Table [number]\n[Description]."
- **Column headers:** Bold or distinct formatting
- **Alignment:** Data aligned appropriately (numbers right-aligned, text left-aligned)

#### Example Captions:
```
Table 1
Dataset characteristics.

Table 2
Attributes in dataset.

Table 3
Statistics of class-wise student distribution in college dataset.
```

#### Table Content Style:
- **Two-column attribute tables:** Attribute | Description
- **Multi-column data tables:** Headers describing each metric
- **No vertical lines** (clean table design)
- **Horizontal lines** only for header separation

### Figure Format:

#### Figure Structure:
- **Figure number and caption BELOW the figure**
- **Caption format:** "Fig. [number]. [Description]."
- **Subfigures:** Labeled as (a), (b), (c), etc.

#### Example Captions:
```
Fig. 1. Decision tree with Accuracy (Board GPA).

Fig. 4. Decision tree with (a) Accuracy, (b) Gini Index, and (c) Information Gain (Proposed GPA-1).

Fig. 6. Workflow of proposed methodology.
```

### Referencing in Text:
- **Tables:** "shown in Table 1", "presented in Table 3"
- **Figures:** "shown in Fig. 6", "shown in Figs. 1, 2, 3, 4, and 5"

---

## 5. CITATION STYLE

### Style: **APA (Author-Year)**

### In-text Citation Patterns:
- **Single author:** (Maxwell, 2012)
- **Multiple authors:** (Kono et al., 2018)
- **Multiple citations:** (Star, 2021)(Tribune, 2020) or (Han et al., 2022)
- **Narrative citation:** According to (Asif et al., 2017)...

### Citation Frequency:
- **High frequency:** 16+ APA-style citations identified
- **Support for claims:** Each major claim backed by citation

### Reference List Format:
- Not fully visible in extracted text
- Alphabetically organized
- Standard APA journal article format expected

---

## 6. WRITING STYLE

### Tone and Voice:
- **Formal academic tone** throughout
- **Third person** perspective
- **Objective language** with passive voice common
- **Present tense** for methodology, past tense for results

### Paragraph Structure:

#### Introduction Paragraphs:
- **Opening sentence:** Broad context statement
- **Middle sentences:** Specific problem or gap
- **Closing sentence:** Study's approach or contribution

#### Methodology Paragraphs:
- **Topic sentence:** What is being described
- **Body:** Technical details with equation support
- **Transition:** Connection to next step

#### Results Paragraphs:
- **Opening:** What is being presented
- **Evidence:** Table/figure references
- **Analysis:** Interpretation of results

### Technical Term Introduction:

#### Pattern:
1. **First mention:** Full term with abbreviation in parentheses
   - Example: "Educational Data Mining (EDM)"
   - Example: "Higher Secondary Certificate (HSC)¹"
2. **Subsequent mentions:** Use abbreviation only
3. **Footnotes:** For contextual explanations (numbered superscripts)

#### Example:
```
¹ Higher Secondary Certificate (HSC) is a public examination in Bangladesh conducted 
by the Boards of Intermediate and Secondary Education under the Ministry of Education.
```

### List Formatting:

#### Bulleted Lists:
- Used for research objectives
- Used for key contributions
- Each bullet point is a complete sentence or phrase
- Parallel structure maintained

#### Numbered Lists:
- Used for sequential steps
- Used for categorized items
- Format: "Firstly,... Secondly,... Thirdly,... Fourthly,..."

### Sentence Construction:
- **Length:** Medium to long (20-30 words average)
- **Complexity:** Compound and complex sentences common
- **Connectors:** "Therefore", "Besides", "Thus", "Accordingly", "Consequently", "Furthermore"

---

## 7. KEY FORMATTING ELEMENTS

### A. Research Objectives Presentation

#### Format:
**Main heading:** "1.1. Research objectives"

**Structure:**
1. Narrative description of each objective
2. **Summary section:** "However, the objectives can be summarized as follows:"
3. **Bulleted list:**
   ```
   • Objective 1: [Brief description]. [Additional context if needed].
   • Objective 2: [Brief description]. [Additional context if needed].
   • Objective 3: [Brief description].
     i) [Sub-objective]
     ii) [Sub-objective]
     iii) [Sub-objective]
   • Objective 4: [Brief description].
   ```

### B. Algorithm/Method Descriptions (Section 3)

#### Structure for each algorithm:
1. **Numbered subsection header** (e.g., "3.1. Decision tree")
2. **Opening definition:** What the algorithm is
3. **Key characteristics:** How it works
4. **Technical details:** Mathematical or procedural explanation
5. **Citations:** Supporting references

#### Example Pattern:
```
3.1. Decision tree

Decision tree is a supervised machine learning algorithm [definition]. 
The tree structure consists of [components] (Citation, Year). 
The algorithm works by [process description]. [Technical details with 
equations or procedures if applicable].
```

### C. Dataset Description

#### Required Tables:

**Table 1: Dataset Characteristics**
- Columns: Dataset name | Attribute type | No. of attributes | No. of instances

**Table 2: Attributes in Dataset**
- Columns: Attribute | Description
- List ALL attributes with detailed descriptions

**Table 3/4: Class Distribution Statistics**
- Columns vary based on classification categories
- Shows distribution across performance categories

### D. Experimental Workflow

#### Components:
1. **Workflow diagram (Figure):** Visual representation
2. **Step-by-step textual description**
3. **Mathematical formulations** for each approach
4. **Variable definitions**
5. **Validation approach** (e.g., k-fold cross-validation)

### E. Results Section Organization

#### Structure:
**Main heading:** "5. Results analysis"

**Introduction paragraph:** Overview of metrics and formulas

**Subsections by objective:**
```
5.1. Objective 1 - [specific title]
5.2. Objective 2 - [specific title]
5.3. Objective 3 - [specific title]
5.4. Objective 4 - [specific title]
```

**Each subsection includes:**
- Results tables
- Interpretation of findings
- Comparison across approaches
- Discussion of implications

### F. Performance Metrics Presentation

#### Format:
1. **Equations first** (numbered, with definitions)
   ```
   𝐴𝑐𝑐𝑢𝑟𝑎𝑐𝑦 = (𝑇𝑃 + 𝑇𝑁) / (𝑇𝑃 + 𝑇𝑁 + 𝐹𝑃 + 𝐹𝑁)     (7)
   ```

2. **Variable definitions:**
   ```
   Here, 𝑇𝑃 denotes True Positive, 𝐹𝑃 denotes False Positive, 
         𝑇𝑁 denotes True Negative and 𝐹𝑁 denotes False Negative.
   ```

3. **Tables showing results:**
   - Table 5: Classifier prediction accuracy
   - Table 6: Classifier kappa
   - Table 7: Classifier prediction weighted F1-score

### G. Comparative Analysis Structure

#### Pattern:
1. **Multiple approaches defined** (Board GPA, Proposed GPA-1, Proposed GPA-2)
2. **Tables comparing all approaches** across multiple classifiers
3. **Analysis of differences** and superiority
4. **Discussion of implications**

---

## 8. ABSTRACT STRUCTURE

### Components (in order):
1. **Context statement** (1-2 sentences)
2. **Problem/Gap identification** (1 sentence)
3. **Study purpose and scope** (1-2 sentences)
4. **Methods overview** (1-2 sentences)
5. **Key findings** (2-3 sentences)
6. **Implications** (1 sentence)

### Keywords:
- **Placement:** Below abstract
- **Format:** "Keywords: [term 1], [term 2], [term 3], [term 4]"
- **Number:** 4-6 keywords
- **Style:** Lowercase except proper nouns

---

## 9. SECTION TRANSITION PATTERN

### Introduction Section Closing:
**Standard format:**
```
"The remainder of the paper is organized into several sections. In 
section 2, [content] are discussed. [Continue for each section]. 
Finally, section [last] concludes our work."
```

### Between Sections:
- No explicit transition paragraphs
- Direct section headers
- First paragraph of new section provides context

---

## 10. SPECIALIZED FORMATTING

### Footnotes:
- **Numbered superscripts** in main text
- **Full explanation at bottom of page** or after paragraph
- Used for contextual information, definitions, or external references

### Emphasis:
- **Italics:** Variables, statistical terms, occasional emphasis
- **Bold:** Section headings, table/figure labels
- **Quotation marks:** For direct quotes or specific terminology

### Abbreviations After Definition:
- **Pattern:** "Full Term (ABBR)" on first use
- Example: "Grade Point Average (GPA)"
- Use only abbreviation thereafter

### Percentage Formatting:
- **With percent sign:** 60%, 50%-59%
- **Ranges:** Use hyphen (0%-49%)

---

## 11. QUALITY MARKERS OF THIS PAPER

### Strengths to Emulate:

1. **Clear objective-driven structure** - Each section explicitly addresses research objectives
2. **Comprehensive methodology** - Detailed explanation with equations, tables, and workflow
3. **Multiple comparison approaches** - Shows rigor through comparative analysis
4. **Visual support** - Decision trees, workflow diagrams
5. **Quantitative rigor** - Multiple evaluation metrics presented
6. **Practical implications** - Connects findings to real-world applications
7. **Technical transparency** - Implementation details provided
8. **Systematic organization** - Logical flow from introduction to conclusion

### Academic Writing Quality:

- **Clarity:** Technical concepts explained accessibly
- **Completeness:** All methodological details provided
- **Coherence:** Strong logical connections between sections
- **Citation density:** Well-supported with relevant literature
- **Reproducibility:** Sufficient detail for replication

---

## 12. LaTeX IMPLEMENTATION NOTES

### Recommended Packages:
- `amsmath` - For equations and mathematical notation
- `graphicx` - For figures
- `booktabs` - For professional tables
- `subcaption` - For subfigures (a), (b), (c)
- `hyperref` - For cross-references
- `natbib` or `biblatex` - For APA citations

### Section Numbering:
- **Level 1:** 1. Section Name
- **Level 2:** 1.1. Subsection Name
- **Level 3:** 1.1.1. Sub-subsection (if needed)

### Equation Environment:
```latex
\begin{equation}
    HSC_{GPA} = (JSC_{GPA} \times 0.25) + (SSC_{GPA} \times 0.75)
    \label{eq:board_gpa}
\end{equation}
```

### Table Environment:
```latex
\begin{table}[h]
\centering
\caption{Dataset characteristics.}
\label{tab:dataset_char}
\begin{tabular}{llll}
\toprule
Dataset name & Attribute type & No. of attributes & No. of instances \\
\midrule
College dataset & Integer & 28 & 309 \\
Synthetic dataset & Integer & 28 & 1000 \\
\bottomrule
\end{tabular}
\end{table}
```

### Figure Environment:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{workflow.png}
\caption{Workflow of proposed methodology.}
\label{fig:workflow}
\end{figure}
```

---

## 13. CONTENT CHECKLIST FOR METHODOLOGY SECTION

When writing your methodology section following this style:

- [ ] Clearly state dataset source and characteristics
- [ ] Provide table of dataset characteristics
- [ ] Provide table of all attributes with descriptions
- [ ] Show class distribution statistics
- [ ] Explain experimental design and motivation
- [ ] List all algorithms/methods used
- [ ] Include workflow diagram
- [ ] Present mathematical formulations with numbered equations
- [ ] Define all variables immediately after equations
- [ ] Specify evaluation metrics with formulas
- [ ] Mention implementation tools/software
- [ ] Describe validation approach (e.g., k-fold cross-validation)
- [ ] Organize by research objectives
- [ ] Provide sufficient detail for reproducibility
- [ ] Connect methodology to research questions

---

## SUMMARY

This reference paper follows a **highly structured, objective-driven approach** with:

- **Clear hierarchical organization** (numbered sections and subsections)
- **Methodology organized by dataset → experimental design → specific objectives**
- **Heavy use of tables** for data presentation
- **Numbered equations** with immediate variable definitions
- **APA citation style** with author-year format
- **Formal academic tone** with objective language
- **Visual aids** (decision trees, workflow diagrams, tables)
- **Comprehensive metric presentation** with mathematical formulas
- **Reproducible details** (tools, parameters, validation methods)

**Key takeaway for your LaTeX document:** Structure your methodology with clear subsections (Dataset, Experimental Methodology), use numbered equations with definitions, provide comprehensive tables, include workflow diagrams, and organize analysis by research objectives.
