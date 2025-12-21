"""
Adaptive Feature Selection Algorithm (AFSA) Implementation
Enhanced feature selection combining ensemble ranking with population-based optimization

Reference: Adaptive Feature Selection Algorithm similar to Fish Swarm/Artificial Fish School Algorithm
Adapted for educational data mining with student dropout prediction
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class AdaptiveFeatureSelector:
    """
    AFSA - Adaptive Feature Selection Algorithm
    
    Combines:
    1. Initial ensemble feature ranking (Information Gain, Gini, Mutual Info, ANOVA, Gain Ratio)
    2. Population-based iterative optimization
    3. Adaptive feature subset exploration based on accuracy feedback
    
    Parameters:
    -----------
    n_features : int
        Total number of features in dataset
    population_size : int
        Number of feature subsets (fish) in the population
    max_iterations : int
        Maximum optimization iterations
    visual_distance : float
        Exploration radius for each fish (0.0-1.0)
    step_size : float
        Movement step size (0.0-1.0)
    crowding_factor : float
        Density threshold for crowding avoidance
    min_features : int
        Minimum features to select
    max_features : int
        Maximum features to select
    random_state : int
        Random seed for reproducibility
    """
    
    def __init__(self, n_features, population_size=20, max_iterations=30,
                 visual_distance=0.3, step_size=0.1, crowding_factor=0.7,
                 min_features=5, max_features=None, random_state=42):
        
        self.n_features = n_features
        self.population_size = population_size
        self.max_iterations = max_iterations
        self.visual_distance = visual_distance
        self.step_size = step_size
        self.crowding_factor = crowding_factor
        self.min_features = min_features
        self.max_features = max_features if max_features else n_features
        self.random_state = random_state
        
        np.random.seed(random_state)
        
        # Population: each fish represents a binary feature subset
        # Shape: (population_size, n_features)
        self.population = None
        self.fitness = None
        self.best_solution = None
        self.best_fitness = -np.inf
        self.fitness_history = []
        
        # Ensemble ranking scores
        self.feature_ranks = None
        
    
    def _ensemble_feature_ranking(self, X, y):
        """
        Phase 1: Ensemble Feature Ranking
        Combines 5 methods: Info Gain, Gini, Mutual Info, ANOVA F-test, Gain Ratio
        """
        print("\n🔍 Phase 1: Ensemble Feature Ranking")
        
        n_features = X.shape[1]
        ensemble_scores = np.zeros(n_features)
        
        # Method 1: Information Gain (Entropy-based)
        print("  Computing Information Gain...")
        rf_ig = RandomForestClassifier(n_estimators=100, criterion='entropy', 
                                       random_state=self.random_state, n_jobs=-1)
        rf_ig.fit(X, y)
        ig_scores = rf_ig.feature_importances_
        ig_ranks = np.argsort(ig_scores)[::-1].argsort()
        
        # Method 2: Gini Importance
        print("  Computing Gini Importance...")
        rf_gini = RandomForestClassifier(n_estimators=100, criterion='gini',
                                         random_state=self.random_state, n_jobs=-1)
        rf_gini.fit(X, y)
        gini_scores = rf_gini.feature_importances_
        gini_ranks = np.argsort(gini_scores)[::-1].argsort()
        
        # Method 3: Mutual Information
        print("  Computing Mutual Information...")
        mi_scores = mutual_info_classif(X, y, random_state=self.random_state)
        mi_ranks = np.argsort(mi_scores)[::-1].argsort()
        
        # Method 4: ANOVA F-statistic
        print("  Computing ANOVA F-test...")
        f_scores, _ = f_classif(X, y)
        f_ranks = np.argsort(f_scores)[::-1].argsort()
        
        # Method 5: Gain Ratio (normalized Information Gain)
        print("  Computing Gain Ratio...")
        # Approximate gain ratio using normalized IG
        intrinsic_values = np.array([self._calculate_intrinsic_value(X[:, i]) for i in range(n_features)])
        gain_ratio_scores = ig_scores / (intrinsic_values + 1e-10)
        gr_ranks = np.argsort(gain_ratio_scores)[::-1].argsort()
        
        # Ensemble: Average rank across all methods
        ensemble_ranks = (ig_ranks + gini_ranks + mi_ranks + f_ranks + gr_ranks) / 5.0
        
        # Convert to scores (lower rank = higher score)
        self.feature_ranks = (n_features - ensemble_ranks) / n_features
        
        print(f"  ✓ Ensemble ranking complete")
        print(f"  Top 10 features by ensemble score: {np.argsort(self.feature_ranks)[::-1][:10]}")
        
        return self.feature_ranks
    
    
    def _calculate_intrinsic_value(self, feature_values):
        """Calculate intrinsic value for gain ratio"""
        unique_vals, counts = np.unique(feature_values, return_counts=True)
        probs = counts / len(feature_values)
        intrinsic = -np.sum(probs * np.log2(probs + 1e-10))
        return intrinsic if intrinsic > 0 else 1.0
    
    
    def _initialize_population(self):
        """
        Phase 2: Initialize Population
        Uses ensemble ranking to bias initial population toward high-ranked features
        """
        print("\n🐟 Phase 2: Initializing AFSA Population")
        
        self.population = np.zeros((self.population_size, self.n_features), dtype=bool)
        
        for i in range(self.population_size):
            # Number of features: random between min and max
            n_selected = np.random.randint(self.min_features, self.max_features + 1)
            
            # Probabilistic selection based on ensemble ranking
            # Higher-ranked features have higher selection probability
            selection_probs = self.feature_ranks / self.feature_ranks.sum()
            
            # Select features without replacement
            selected_features = np.random.choice(
                self.n_features, 
                size=n_selected, 
                replace=False,
                p=selection_probs
            )
            
            self.population[i, selected_features] = True
        
        print(f"  ✓ Initialized {self.population_size} feature subsets")
        print(f"  Feature count range: {self.population.sum(axis=1).min()}-{self.population.sum(axis=1).max()}")
    
    
    def _evaluate_fitness(self, X, y, model):
        """
        Evaluate fitness of all population members
        Fitness = Cross-validated accuracy on training data
        """
        self.fitness = np.zeros(self.population_size)
        
        for i in range(self.population_size):
            selected_features = self.population[i]
            
            if selected_features.sum() == 0:
                self.fitness[i] = 0.0
                continue
            
            X_subset = X[:, selected_features]
            
            # 3-fold CV for speed (balance between reliability and computation)
            cv_scores = cross_val_score(model, X_subset, y, cv=3, 
                                       scoring='accuracy', n_jobs=-1)
            self.fitness[i] = cv_scores.mean()
        
        return self.fitness
    
    
    def _prey_behavior(self, fish_idx, X, y, model):
        """
        Prey Behavior: Move toward better feature subset in visual range
        """
        current_subset = self.population[fish_idx].copy()
        current_fitness = self.fitness[fish_idx]
        
        # Explore visual range: flip random features
        n_flips = max(1, int(self.n_features * self.visual_distance))
        trial_subset = current_subset.copy()
        
        # Randomly flip features (add or remove)
        flip_indices = np.random.choice(self.n_features, size=n_flips, replace=False)
        trial_subset[flip_indices] = ~trial_subset[flip_indices]
        
        # Ensure within min/max constraints
        n_selected = trial_subset.sum()
        if n_selected < self.min_features:
            # Add random features
            available = ~trial_subset
            add_count = self.min_features - n_selected
            add_indices = np.random.choice(np.where(available)[0], size=add_count, replace=False)
            trial_subset[add_indices] = True
        elif n_selected > self.max_features:
            # Remove random features
            remove_count = n_selected - self.max_features
            remove_indices = np.random.choice(np.where(trial_subset)[0], size=remove_count, replace=False)
            trial_subset[remove_indices] = False
        
        # Evaluate trial subset
        if trial_subset.sum() > 0:
            X_trial = X[:, trial_subset]
            trial_fitness = cross_val_score(model, X_trial, y, cv=3, 
                                           scoring='accuracy', n_jobs=-1).mean()
            
            # Move if better
            if trial_fitness > current_fitness:
                return trial_subset, trial_fitness
        
        return current_subset, current_fitness
    
    
    def _swarm_behavior(self, fish_idx, X, y, model):
        """
        Swarm Behavior: Move toward center of nearby high-fitness fish
        """
        current_subset = self.population[fish_idx].copy()
        current_fitness = self.fitness[fish_idx]
        
        # Find nearby fish with better fitness
        better_fish = self.fitness > current_fitness
        
        if better_fish.sum() == 0:
            return current_subset, current_fitness
        
        # Compute center of better fish (feature-wise majority voting)
        center_subset = self.population[better_fish].mean(axis=0) > 0.5
        
        # Move partially toward center
        step = np.random.rand(self.n_features) < self.step_size
        trial_subset = current_subset.copy()
        trial_subset[step] = center_subset[step]
        
        # Ensure constraints
        n_selected = trial_subset.sum()
        if n_selected < self.min_features or n_selected > self.max_features:
            return current_subset, current_fitness
        
        # Evaluate
        if trial_subset.sum() > 0:
            X_trial = X[:, trial_subset]
            trial_fitness = cross_val_score(model, X_trial, y, cv=3,
                                           scoring='accuracy', n_jobs=-1).mean()
            
            if trial_fitness > current_fitness:
                return trial_subset, trial_fitness
        
        return current_subset, current_fitness
    
    
    def _follow_behavior(self, fish_idx, X, y, model):
        """
        Follow Behavior: Move toward best fish in population
        """
        current_subset = self.population[fish_idx].copy()
        current_fitness = self.fitness[fish_idx]
        
        # Find best fish
        best_idx = np.argmax(self.fitness)
        best_subset = self.population[best_idx]
        
        if best_idx == fish_idx:
            return current_subset, current_fitness
        
        # Move toward best (partial adoption of best features)
        step = np.random.rand(self.n_features) < self.step_size
        trial_subset = current_subset.copy()
        trial_subset[step] = best_subset[step]
        
        # Ensure constraints
        n_selected = trial_subset.sum()
        if n_selected < self.min_features or n_selected > self.max_features:
            return current_subset, current_fitness
        
        # Evaluate
        if trial_subset.sum() > 0:
            X_trial = X[:, trial_subset]
            trial_fitness = cross_val_score(model, X_trial, y, cv=3,
                                           scoring='accuracy', n_jobs=-1).mean()
            
            if trial_fitness > current_fitness:
                return trial_subset, trial_fitness
        
        return current_subset, current_fitness
    
    
    def fit(self, X, y, model=None):
        """
        Execute AFSA optimization
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target labels
        model : sklearn classifier
            Base model for fitness evaluation (default: RandomForest)
        """
        if model is None:
            model = RandomForestClassifier(n_estimators=100, random_state=self.random_state, n_jobs=-1)
        
        print("="*80)
        print("  ADAPTIVE FEATURE SELECTION ALGORITHM (AFSA)")
        print("="*80)
        
        # Phase 1: Ensemble Ranking
        self._ensemble_feature_ranking(X, y)
        
        # Phase 2: Initialize Population
        self._initialize_population()
        
        # Phase 3: Iterative Optimization
        print(f"\n🔄 Phase 3: Iterative Optimization ({self.max_iterations} iterations)")
        
        for iteration in range(self.max_iterations):
            # Evaluate all fish
            self._evaluate_fitness(X, y, model)
            
            # Track best
            iter_best_idx = np.argmax(self.fitness)
            iter_best_fitness = self.fitness[iter_best_idx]
            
            if iter_best_fitness > self.best_fitness:
                self.best_fitness = iter_best_fitness
                self.best_solution = self.population[iter_best_idx].copy()
            
            self.fitness_history.append(self.best_fitness)
            
            # Progress report every 5 iterations
            if (iteration + 1) % 5 == 0 or iteration == 0:
                print(f"  Iteration {iteration+1:2d}/{self.max_iterations}: "
                      f"Best Fitness = {self.best_fitness:.4f} "
                      f"({self.best_solution.sum()} features)")
            
            # Update each fish
            for i in range(self.population_size):
                # Randomly choose behavior
                behavior = np.random.choice(['prey', 'swarm', 'follow'], p=[0.4, 0.3, 0.3])
                
                if behavior == 'prey':
                    new_subset, new_fitness = self._prey_behavior(i, X, y, model)
                elif behavior == 'swarm':
                    new_subset, new_fitness = self._swarm_behavior(i, X, y, model)
                else:  # follow
                    new_subset, new_fitness = self._follow_behavior(i, X, y, model)
                
                self.population[i] = new_subset
                self.fitness[i] = new_fitness
        
        print(f"\n✅ Optimization Complete!")
        print(f"  Best Accuracy: {self.best_fitness:.4f}")
        print(f"  Optimal Features: {self.best_solution.sum()}/{self.n_features}")
        print(f"  Selected Feature Indices: {np.where(self.best_solution)[0]}")
        
        return self
    
    
    def transform(self, X):
        """Apply selected features to dataset"""
        if self.best_solution is None:
            raise ValueError("Must call fit() before transform()")
        return X[:, self.best_solution]
    
    
    def fit_transform(self, X, y, model=None):
        """Fit and transform in one step"""
        self.fit(X, y, model)
        return self.transform(X)
    
    
    def get_selected_features(self):
        """Return indices of selected features"""
        if self.best_solution is None:
            raise ValueError("Must call fit() before get_selected_features()")
        return np.where(self.best_solution)[0]
    
    
    def get_feature_mask(self):
        """Return boolean mask of selected features"""
        return self.best_solution


def demo_afsa():
    """
    Demonstration of AFSA on sample educational data
    """
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report
    
    print("\n" + "="*80)
    print("  AFSA DEMONSTRATION: Educational Dropout Prediction")
    print("="*80)
    
    # Simulate educational dataset (similar to thesis data)
    X, y = make_classification(
        n_samples=1000,
        n_features=46,
        n_informative=20,
        n_redundant=10,
        n_classes=2,
        class_sep=1.2,
        random_state=42
    )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    print(f"\nDataset: {X_train.shape[0]} training samples, {X.shape[1]} features")
    
    # Apply AFSA
    afsa = AdaptiveFeatureSelector(
        n_features=X.shape[1],
        population_size=15,
        max_iterations=20,
        min_features=10,
        max_features=30,
        random_state=42
    )
    
    X_train_selected = afsa.fit_transform(X_train, y_train)
    X_test_selected = afsa.transform(X_test)
    
    print(f"\n📊 Feature Selection Results:")
    print(f"  Original features: {X.shape[1]}")
    print(f"  Selected features: {X_train_selected.shape[1]}")
    print(f"  Reduction: {(1 - X_train_selected.shape[1]/X.shape[1])*100:.1f}%")
    
    # Compare performance
    print(f"\n🎯 Model Performance Comparison:")
    
    # Model with all features
    rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_all.fit(X_train, y_train)
    y_pred_all = rf_all.predict(X_test)
    acc_all = accuracy_score(y_test, y_pred_all)
    
    # Model with AFSA-selected features
    rf_afsa = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_afsa.fit(X_train_selected, y_train)
    y_pred_afsa = rf_afsa.predict(X_test_selected)
    acc_afsa = accuracy_score(y_test, y_pred_afsa)
    
    print(f"  All Features (46):        Accuracy = {acc_all:.4f}")
    print(f"  AFSA Features ({X_train_selected.shape[1]}):       Accuracy = {acc_afsa:.4f}")
    print(f"  Improvement:              {(acc_afsa - acc_all)*100:+.2f}%")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    demo_afsa()
