# Student Performance and Dropout Prediction System

## Research Project: Educational Data Mining with Deep Learning and LLM

This project implements a **journal-quality** student performance prediction, dropout risk assessment, and personalized recommendation system using deep learning and Large Language Models (LLMs).

**✨ NEW**: Journal methodology implementation with **real dataset (4,424 students)**

---

## 📋 Project Overview

**Research Title**: Student Performance and Dropout Prediction using Deep Learning and LLM

**Publication Target**: IEEE Transactions on Learning Technologies, Computers & Education

**Key Features**:
- 🎯 3-class outcome prediction (Graduate/Enrolled/Dropout)
- ⚠️ Binary dropout risk prediction with attention mechanism
- 🔬 Multi-task learning architecture
- 🤖 LLM-powered personalized interventions
- 📊 Comprehensive evaluation with publication-quality visualizations
- 🧠 State-of-the-art deep learning architectures
- 📈 Feature engineering following educational research best practices

**Dataset**:
- **Real Dataset**: 4,424 students, 35 features (demographic, academic, socioeconomic, macroeconomic)
- **Mock Dataset**: 50 students, 31 features (for prototyping)

---

## 📁 Project Structure

```
Final Thesis project/
│
├── data/
│   ├── ulab_students_dataset.csv       # Mock dataset (50 students)
│   └── processed/                      # Processed data (generated)
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py           # Data cleaning and feature engineering
│   ├── models/
│   │   ├── __init__.py
│   │   ├── performance_model.py        # Performance prediction DNN
│   │   ├── dropout_model.py            # Dropout prediction DNN
│   │   └── hybrid_model.py             # Multi-task learning model
│   ├── llm/
│   │   ├── __init__.py
│   │   └── recommendation_engine.py    # LLM-based recommendations
│   ├── evaluation.py                   # Model evaluation metrics
│   └── visualization.py                # Plotting and visualization
│
├── notebooks/
│   ├── 01_data_exploration.ipynb       # EDA notebook
│   ├── 02_model_training.ipynb         # Model training experiments
│   └── 03_recommendations.ipynb        # LLM recommendation testing
│
├── docs/
│   └── METHODOLOGY.md                  # Detailed methodology
│
├── outputs/
│   ├── models/                         # Saved trained models
│   ├── plots/                          # Generated visualizations
│   └── reports/                        # Student reports
│
├── main.py                             # Main execution script
├── requirements.txt                    # Python dependencies
├── .env.example                        # Environment variables template
└── README.md                           # This file
```

---

## 🚀 Installation

### 1. Clone the repository (or navigate to project folder)
```bash
cd "d:\MS program\Final Thesis\Final Thesis project"
```

### 2. Create a virtual environment
```bash
python -m venv venv
```

### 3. Activate virtual environment
**Windows:**
```bash
.\venv\Scripts\Activate.ps1
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Set up environment variables
```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

---

## 📊 Dataset Description

**Mock Dataset**: 50 ULAB undergraduate students with 31 features

**Student Population**: Undergraduate students (Semesters 1-8) from various departments

### Features:
- **Demographics**: Age, Gender, Department, Semester
- **Academic**: CGPA, Attendance, Scores, Submission rates
- **Behavioral**: Study hours, Sleep, Social media usage, Stress
- **Socioeconomic**: Family income, Parents' education, Distance
- **Support**: Scholarship, Mentor meetings, Health issues

### Target Variables:
- `dropout_status`: Yes/No (binary classification)
- `final_grade`: A+, A, A-, B+, B, B-, C+, C, D+ (multi-class)

---

## 🧠 Methodology

### Phase 1: Data Preprocessing
- Data cleaning and imputation
- Feature engineering (derived features)
- Encoding and normalization
- Train-test split (70-15-15)

### Phase 2: Deep Learning Models

#### Model 1: Performance Prediction
- Architecture: 4-layer DNN
- Activation: ReLU, Softmax
- Regularization: Dropout, BatchNorm
- Output: Grade classification

#### Model 2: Dropout Prediction
- Architecture: DNN with Attention
- Activation: ReLU, Sigmoid
- Output: Binary (dropout risk)

#### Model 3: Hybrid Multi-Task Model
- Shared feature extraction
- Two prediction heads
- Joint optimization

### Phase 3: LLM Recommendations
- Student profile aggregation
- Prompt engineering
- OpenAI GPT-4 integration
- Personalized actionable recommendations

### Phase 4: Evaluation
- Metrics: Accuracy, F1, Precision, Recall, AUC-ROC
- 5-Fold Cross-Validation
- Confusion matrices
- Feature importance analysis

---

## 💻 Usage

### Run the complete pipeline:
```bash
python main.py
```

### Run specific modules:

**Data Preprocessing:**
```python
from src.data_preprocessing import DataPreprocessor

preprocessor = DataPreprocessor('data/ulab_students_dataset.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_data()
```

**Train Performance Model:**
```python
from src.models.performance_model import PerformanceModel

model = PerformanceModel(input_dim=X_train.shape[1])
model.train(X_train, y_train, epochs=100)
predictions = model.predict(X_test)
```

**Generate Recommendations:**
```python
from src.llm.recommendation_engine import RecommendationEngine

engine = RecommendationEngine(api_key='your-openai-key')
recommendations = engine.generate_recommendations(student_profile)
```

---

## 📈 Expected Results

### Model Performance Targets:
- **Performance Prediction**: >85% accuracy
- **Dropout Prediction**: >90% accuracy, >0.85 AUC-ROC

### Outputs:
1. Trained model files in `outputs/models/`
2. Visualization plots in `outputs/plots/`
3. Student recommendation reports in `outputs/reports/`
4. Comprehensive evaluation metrics

---

## 🔬 Research Contributions

1. **Multi-task deep learning** for educational data
2. **LLM integration** for interpretable recommendations
3. **Attention mechanisms** for feature importance
4. **Proactive intervention framework** for at-risk students

---

## 📝 Citation

If you use this work, please cite:

```
@mastersthesis{ulab_student_prediction,
  title={Student Performance and Dropout Prediction and Personalized Recommendation using Deep Learning and LLM of ULAB Students},
  author={Your Name},
  year={2025},
  school={Your University}
}
```

---

## 🤝 Contributing

This is a research project. For suggestions or improvements:
1. Document your changes
2. Test thoroughly
3. Update documentation

---

## 📧 Contact

For questions or collaboration:
- Email: your.email@example.com
- Institution: University of Liberal Arts Bangladesh (ULAB)

---

## 🔒 License

This project is for academic research purposes.

---

## ⚠️ Ethical Considerations

- **Privacy**: All student data is anonymized
- **Fairness**: Models evaluated for demographic bias
- **Transparency**: Explainable AI methods used
- **Human Oversight**: Recommendations require faculty review

---

## 🚧 Future Work

- [ ] Scale to 1000+ students
- [ ] Real-time monitoring dashboard
- [ ] Mobile application
- [ ] Temporal analysis (time-series)
- [ ] Feedback loop integration
- [ ] Deployment to production environment

---

**Last Updated**: November 2025
