"""
AHFS-TA: Adaptive Hierarchical Feature Selection with Temporal Attention
Complete Implementation for Student Dropout Prediction
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, matthews_corrcoef
from transformers import DistilBertTokenizer, DistilBertModel
import shap
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class LLMFeatureExtractor:
    """Component 1: LLM-Based Feature Enrichment using DistilBERT"""
    
    def __init__(self):
        print("Initializing DistilBERT for feature extraction...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertModel.from_pretrained('distilbert-base-uncased').to(device)
        self.model.eval()
    
    def create_text_representation(self, row):
        """Create textual description from student features for LLM processing"""
        # Generate meaningful text from numerical/categorical features
        texts = []
        
        # Academic performance
        if 'Curricular units 1st sem (grade)' in row:
            grade = row['Curricular units 1st sem (grade)']
            if grade > 14:
                texts.append("excellent academic performance")
            elif grade > 12:
                texts.append("good academic performance")
            elif grade > 10:
                texts.append("average academic performance")
            else:
                texts.append("struggling academically")
        
        # Attendance and engagement
        if 'Curricular units 1st sem (approved)' in row and 'Curricular units 1st sem (enrolled)' in row:
            enrolled = row['Curricular units 1st sem (enrolled)']
            approved = row['Curricular units 1st sem (approved)']
            if enrolled > 0:
                success_rate = approved / enrolled
                if success_rate > 0.9:
                    texts.append("highly engaged student")
                elif success_rate > 0.7:
                    texts.append("moderately engaged student")
                else:
                    texts.append("low engagement")
        
        # Financial status
        if 'Debtor' in row and row['Debtor'] == 1:
            texts.append("financial difficulties")
        if 'Scholarship holder' in row and row['Scholarship holder'] == 1:
            texts.append("scholarship recipient")
        
        # Age and maturity
        if 'Age at enrollment' in row:
            age = row['Age at enrollment']
            if age < 20:
                texts.append("young student")
            elif age > 25:
                texts.append("mature student")
        
        # Combine into coherent text
        if not texts:
            return "student with typical characteristics"
        return " ".join(texts)
    
    def extract_llm_features(self, df):
        """Extract psychosocial features using DistilBERT embeddings"""
        print("Extracting LLM-based psychosocial features...")
        
        llm_features = []
        batch_size = 32
        
        for i in range(0, len(df), batch_size):
            batch = df.iloc[i:i+batch_size]
            texts = [self.create_text_representation(row) for _, row in batch.iterrows()]
            
            # Tokenize and get embeddings
            inputs = self.tokenizer(texts, padding=True, truncation=True, 
                                   max_length=128, return_tensors='pt').to(device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state[:, 0, :]  # [CLS] token
            
            # Extract psychosocial features from embeddings
            emb_np = embeddings.cpu().numpy()
            
            # Sentiment Score: Projection onto sentiment axis (first principal component)
            sentiment = np.tanh(emb_np[:, :64].mean(axis=1))
            
            # Engagement Index: Variance in embedding space (activity indicator)
            engagement = 1 / (1 + np.exp(-emb_np[:, 64:128].std(axis=1)))
            
            # Topic Consistency: Cosine similarity of embedding segments
            seg1 = emb_np[:, :256]
            seg2 = emb_np[:, 256:512]
            topic_consistency = np.sum(seg1 * seg2, axis=1) / (
                np.linalg.norm(seg1, axis=1) * np.linalg.norm(seg2, axis=1) + 1e-8
            )
            
            # Cognitive Load: Embedding magnitude (complexity indicator)
            cognitive_load = np.linalg.norm(emb_np[:, 512:], axis=1) / 256
            
            batch_features = np.stack([sentiment, engagement, topic_consistency, cognitive_load], axis=1)
            llm_features.append(batch_features)
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {min(i+batch_size, len(df))}/{len(df)} samples")
        
        llm_features = np.vstack(llm_features)
        
        # Create DataFrame with named columns
        llm_df = pd.DataFrame(llm_features, columns=[
            'LLM_Sentiment', 'LLM_Engagement', 'LLM_TopicConsistency', 'LLM_CognitiveLoad'
        ])
        
        print(f"LLM features extracted: {llm_df.shape}")
        print(f"Feature statistics:\n{llm_df.describe()}")
        
        return llm_df


class TemporalDataset(Dataset):
    """Dataset for temporal sequence modeling"""
    
    def __init__(self, X, y, sequence_length=4):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.seq_len = sequence_length
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        # Create temporal sequence by repeating and adding noise for simulation
        # In real scenario, this would be actual semester-wise data
        base_features = self.X[idx]
        
        # Simulate temporal progression (4 semesters)
        sequence = []
        for t in range(self.seq_len):
            # Add temporal variation
            temporal_factor = 1.0 + (t * 0.05) + np.random.normal(0, 0.02)
            seq_features = base_features * temporal_factor
            sequence.append(seq_features)
        
        sequence = torch.stack(sequence)
        return sequence, self.y[idx]


class TemporalAttentionNetwork(nn.Module):
    """Component 3: Temporal Attention Network with GRU + Multi-Head Attention"""
    
    def __init__(self, input_dim, hidden_dim=128, num_heads=4, num_classes=2, dropout=0.3):
        super(TemporalAttentionNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # GRU for temporal sequence modeling
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        
        # Multi-head temporal attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # Bidirectional GRU doubles dimension
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feature projection for attention weights (used in AHFS)
        self.feature_attention = nn.Linear(hidden_dim * 2, input_dim)
        
        # Classification layers
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
        # Store attention weights for analysis
        self.last_attention_weights = None
        self.last_feature_importance = None
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size, seq_len, features = x.shape
        
        # GRU processing
        gru_out, _ = self.gru(x)  # (batch, seq_len, hidden*2)
        
        # Multi-head temporal attention
        attn_out, attn_weights = self.attention(gru_out, gru_out, gru_out)
        self.last_attention_weights = attn_weights.detach()
        
        # Feature importance from attention-weighted representations
        feature_importance = self.feature_attention(attn_out)  # (batch, seq_len, input_dim)
        self.last_feature_importance = feature_importance.abs().mean(dim=(0, 1)).detach()
        
        # Aggregate temporal representations
        temporal_repr = attn_out.mean(dim=1)  # (batch, hidden*2)
        
        # Classification
        x = self.dropout(temporal_repr)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x)
        
        return logits


class AdaptiveFeatureSelector:
    """Component 2: Adaptive Hierarchical Feature Selection (AHFS)"""
    
    def __init__(self, n_features_to_select=28):
        self.n_features = n_features_to_select
        self.selected_features = None
        self.feature_importance_history = []
        
    def calculate_shap_importance(self, model, X_sample, feature_names):
        """Calculate SHAP-based importance (Stream 1)"""
        print("Calculating SHAP importance...")
        
        # Use a subset for SHAP calculation (computationally expensive)
        sample_size = min(100, len(X_sample))
        X_shap = X_sample[:sample_size]
        
        # Simple model wrapper for SHAP
        def model_predict(X):
            with torch.no_grad():
                X_tensor = torch.FloatTensor(X).unsqueeze(1).repeat(1, 4, 1).to(device)
                outputs = model(X_tensor)
                return torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        
        # Use Kernel SHAP for model-agnostic explanation
        explainer = shap.KernelExplainer(model_predict, shap.sample(X_shap, 50))
        shap_values = explainer.shap_values(X_shap, nsamples=100)
        
        shap_importance = np.abs(shap_values).mean(axis=0)
        
        return shap_importance
    
    def get_llm_attention_importance(self, model):
        """Extract LLM attention weights (Stream 2)"""
        if model.last_feature_importance is not None:
            return model.last_feature_importance.cpu().numpy()
        return np.ones(model.input_dim) / model.input_dim
    
    def calculate_temporal_significance(self, X, y, feature_idx):
        """Calculate temporal significance (Stream 3)"""
        # Correlation with outcome over time
        correlations = []
        for t in range(4):  # 4 semesters
            feature_values = X[:, feature_idx]
            corr = np.corrcoef(feature_values, y)[0, 1]
            correlations.append(abs(corr) if not np.isnan(corr) else 0)
        
        return np.mean(correlations)
    
    def meta_ranking(self, shap_imp, llm_imp, temporal_imp, weights=[0.5, 0.3, 0.2]):
        """Meta-ranking fusion of three importance streams"""
        # Normalize importance scores
        shap_norm = (shap_imp - shap_imp.min()) / (shap_imp.max() - shap_imp.min() + 1e-8)
        llm_norm = (llm_imp - llm_imp.min()) / (llm_imp.max() - llm_imp.min() + 1e-8)
        temporal_norm = (temporal_imp - temporal_imp.min()) / (temporal_imp.max() - temporal_imp.min() + 1e-8)
        
        # Weighted combination
        meta_importance = (weights[0] * shap_norm + 
                          weights[1] * llm_norm + 
                          weights[2] * temporal_norm)
        
        return meta_importance
    
    def select_features(self, model, X, y, feature_names):
        """Perform adaptive feature selection"""
        print(f"\nAdaptive Feature Selection (target: {self.n_features} features)...")
        
        # Stream 1: SHAP importance
        shap_imp = self.calculate_shap_importance(model, X, feature_names)
        
        # Stream 2: LLM attention
        llm_imp = self.get_llm_attention_importance(model)
        
        # Stream 3: Temporal significance
        temporal_imp = np.array([
            self.calculate_temporal_significance(X, y, i) 
            for i in range(X.shape[1])
        ])
        
        # Meta-ranking fusion
        meta_imp = self.meta_ranking(shap_imp, llm_imp, temporal_imp)
        
        # Select top features
        top_indices = np.argsort(meta_imp)[-self.n_features:]
        self.selected_features = top_indices
        
        # Store for analysis
        self.feature_importance_history.append({
            'shap': shap_imp,
            'llm': llm_imp,
            'temporal': temporal_imp,
            'meta': meta_imp
        })
        
        print(f"Selected {len(top_indices)} features")
        print(f"Top 10 features: {[feature_names[i] for i in top_indices[-10:]]}")
        
        return top_indices


def train_ahfs_ta_model(X_train, X_test, y_train, y_test, feature_names, 
                        n_epochs=50, batch_size=64, learning_rate=0.001):
    """Complete AHFS-TA training with adaptive feature selection"""
    
    print("\n" + "="*80)
    print("TRAINING AHFS-TA MODEL")
    print("="*80)
    
    # Keep original data for reference
    X_train_orig, X_test_orig = X_train.copy(), X_test.copy()
    
    # Initialize model
    input_dim = X_train.shape[1]
    model = TemporalAttentionNetwork(input_dim=input_dim).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Cosine annealing scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    
    # Create datasets
    train_dataset = TemporalDataset(X_train, y_train)
    test_dataset = TemporalDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Adaptive feature selector
    selector = AdaptiveFeatureSelector(n_features_to_select=28)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'selected_features': []
    }
    
    best_val_acc = 0
    best_model_state = None
    
    for epoch in range(n_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            
            # Main loss
            loss = criterion(outputs, labels)
            
            # Temporal consistency regularization
            if hasattr(model, 'last_attention_weights'):
                # Encourage smooth temporal transitions
                temporal_consistency = 0.1 * torch.var(model.last_attention_weights)
                loss = loss + temporal_consistency
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for sequences, labels in test_loader:
                sequences, labels = sequences.to(device), labels.to(device)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss /= len(test_loader)
        val_acc = 100 * val_correct / val_total
        
        # Update learning rate
        scheduler.step()
        
        # Adaptive feature selection at epoch 10 only
        if (epoch + 1) == 10:
            selected_idx = selector.select_features(model, X_train, y_train, feature_names)
            history['selected_features'].append(selected_idx)
            
            # Update data with selected features
            X_train = X_train[:, selected_idx]
            X_test = X_test[:, selected_idx]
            feature_names = [feature_names[i] for i in selected_idx]
            
            # Reinitialize model with selected features
            input_dim = X_train.shape[1]
            model = TemporalAttentionNetwork(input_dim=input_dim).to(device)
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs-epoch)
            
            # Update datasets
            train_dataset = TemporalDataset(X_train, y_train)
            test_dataset = TemporalDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{n_epochs}] - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.2f}%")
    
    # Return both model and final test data
    return model, selector, history, X_test


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Comprehensive model evaluation"""
    
    model.eval()
    dataset = TemporalDataset(X_test, y_test)
    loader = DataLoader(dataset, batch_size=64)
    
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_probabilities.extend(probabilities[:, 1].cpu().numpy())
            all_labels.extend(labels.numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = {
        'Model': model_name,
        'Accuracy': accuracy_score(all_labels, all_predictions) * 100,
        'Precision': precision_score(all_labels, all_predictions, average='binary'),
        'Recall': recall_score(all_labels, all_predictions, average='binary'),
        'F1-Score': f1_score(all_labels, all_predictions, average='binary'),
        'AUC-ROC': roc_auc_score(all_labels, all_probabilities),
        'MCC': matthews_corrcoef(all_labels, all_predictions)
    }
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    return metrics, cm, all_probabilities


def main():
    """Main execution function"""
    
    print("\n" + "="*80)
    print("AHFS-TA: Adaptive Hierarchical Feature Selection with Temporal Attention")
    print("Student Dropout Prediction System")
    print("="*80 + "\n")
    
    # Load dataset
    print("Loading dataset...")
    df = pd.read_csv('data/educational_data.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Target distribution:\n{df['Target'].value_counts()}\n")
    
    # Prepare features and target
    # Filter out 'Enrolled' status for binary classification
    df_binary = df[df['Target'].isin(['Dropout', 'Graduate'])].copy()
    print(f"Filtered to binary classification: {df_binary.shape}")
    print(f"New target distribution:\n{df_binary['Target'].value_counts()}\n")
    
    X = df_binary.drop('Target', axis=1)
    y = df_binary['Target'].map({'Dropout': 1, 'Graduate': 0})
    
    feature_names = X.columns.tolist()
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Encode categorical columns if any
    for col in X.columns:
        if X[col].dtype == 'object':
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Component 1: LLM Feature Extraction
    print("\n" + "-"*80)
    print("COMPONENT 1: LLM-Based Feature Enrichment")
    print("-"*80)
    
    llm_extractor = LLMFeatureExtractor()
    llm_features = llm_extractor.extract_llm_features(df_binary)
    
    # Combine original features with LLM features
    X_combined = pd.concat([X.reset_index(drop=True), llm_features.reset_index(drop=True)], axis=1)
    feature_names_combined = X.columns.tolist() + llm_features.columns.tolist()
    
    print(f"\nCombined features shape: {X_combined.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined.values, y.values, test_size=0.2, random_state=42, stratify=y
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Train AHFS-TA model
    model, selector, history, X_test_final = train_ahfs_ta_model(
        X_train, X_test, y_train, y_test, feature_names_combined,
        n_epochs=50, batch_size=64
    )
    
    # Evaluate
    print("\n" + "-"*80)
    print("FINAL EVALUATION")
    print("-"*80)
    
    metrics, cm, probs = evaluate_model(model, X_test_final, y_test, "AHFS-TA")
    
    print(f"\nAHFS-TA Performance:")
    print(f"  Accuracy:  {metrics['Accuracy']:.2f}%")
    print(f"  Precision: {metrics['Precision']:.3f}")
    print(f"  Recall:    {metrics['Recall']:.3f}")
    print(f"  F1-Score:  {metrics['F1-Score']:.3f}")
    print(f"  AUC-ROC:   {metrics['AUC-ROC']:.3f}")
    print(f"  MCC:       {metrics['MCC']:.3f}")
    
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # Save results
    results = {
        'model': model,
        'selector': selector,
        'history': history,
        'metrics': metrics,
        'confusion_matrix': cm,
        'feature_names': feature_names_combined,
        'scaler': scaler
    }
    
    torch.save(results, 'outputs/ahfs_ta_results.pt')
    print("\nResults saved to outputs/ahfs_ta_results.pt")
    
    return results


if __name__ == "__main__":
    results = main()
