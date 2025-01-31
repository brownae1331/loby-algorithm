import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from typing import Dict, List, Tuple


class ModelAnalyzer:
    def __init__(self, model):
        """
        Initialize ModelAnalyzer with a trained XGBoost model
        Args:
            model: Trained XGBoost model
        """
        self.model = model
        self.feature_names = [
            "budget_overlap",
            "viewer_age",
            "candidate_age",
            "age_difference",
            "origin_country_match",
            "viewer_country_code",
            "candidate_country_code",
            "course_match",
            "viewer_course_code",
            "candidate_course_code",
            "university_match",
            "viewer_university_code",
            "candidate_university_code",
            "occupation_match",
            "viewer_occupation_code",
            "candidate_occupation_code",
            "industry_match",
            "viewer_industry_code",
            "candidate_industry_code",
            "smoking_match",
            "viewer_smoking_code",
            "candidate_smoking_code",
            "activity_hours_match",
            "viewer_activity_code",
            "candidate_activity_code",
            "viewer_extrovert",
            "candidate_extrovert",
            "extrovert_difference",
            "viewer_cleanliness",
            "candidate_cleanliness",
            "cleanliness_difference",
            "viewer_partying",
            "candidate_partying",
            "partying_difference",
            "gender_match",
            "viewer_gender_code",
            "candidate_gender_code",
            "orientation_match",
            "viewer_orientation_code",
            "candidate_orientation_code",
            "language_overlap",
            "viewer_language_count",
            "candidate_language_count",
            "has_english_match",
        ]

    def analyze_feature_importance(self, plot: bool = True):
        """
        Analyze and visualize which features contribute most to matches
        """
        importance = self.model.get_booster().get_score(importance_type="gain")

        # Convert feature indices to feature names
        named_importance = {}
        for idx, score in importance.items():
            feature_idx = int(idx.replace("f", ""))
            if feature_idx < len(self.feature_names):
                named_importance[self.feature_names[feature_idx]] = score

        # Sort by importance score
        sorted_importance = sorted(
            named_importance.items(), key=lambda x: x[1], reverse=True
        )

        print("\nFeature Importance Analysis:")
        print("-" * 50)
        for feature, score in sorted_importance:
            print(f"{feature}: {score:.4f}")

        if plot:
            plt.figure(figsize=(12, 8))
            features, scores = zip(*sorted_importance)
            plt.barh(features, scores)
            plt.title("Feature Importance")
            plt.xlabel("Importance Score")
            plt.tight_layout()
            plt.show()

    def evaluate_model(self, X_test, y_test):
        """
        Comprehensive model evaluation with metrics and visualizations
        """
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)

        # Calculate metrics
        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
        }

        print("\nModel Performance Metrics:")
        print("-" * 50)
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        # Plot confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.show()

    def analyze_matching_patterns(self, X_test, y_test):
        """
        Analyze patterns in successful and unsuccessful matches
        """
        X_test_np = np.array(X_test)
        successful_matches = X_test_np[y_test == 1]
        unsuccessful_matches = X_test_np[y_test == 0]

        print("\nMatching Pattern Analysis:")
        print("-" * 50)

        # Analyze key features in successful matches
        print("\nSuccessful Matches Pattern:")
        self._analyze_feature_patterns(successful_matches)

        # Compare with unsuccessful matches
        print("\nUnsuccessful Matches Pattern:")
        self._analyze_feature_patterns(unsuccessful_matches)

    def _analyze_feature_patterns(self, matches):
        """Helper method to analyze patterns in matches"""
        if len(matches) == 0:
            print("No matches to analyze")
            return

        # Calculate mean values for key features
        feature_means = np.mean(matches, axis=0)
        feature_stds = np.std(matches, axis=0)

        key_features = {
            "budget_overlap": 0,
            "age_difference": 3,
            "origin_country_match": 4,
            "course_match": 7,
            "occupation_match": 13,
            "smoking_match": 19,
            "activity_hours_match": 22,
        }

        for feature_name, idx in key_features.items():
            print(f"{feature_name}:")
            print(f"  Mean: {feature_means[idx]:.2f}")
            print(f"  Std: {feature_stds[idx]:.2f}")

    def find_strong_correlations(self, X_test):
        """
        Find strong correlations between features
        """
        corr_matrix = np.corrcoef(np.array(X_test).T)
        strong_correlations = []

        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i][j]) > 0.5:  # Threshold for strong correlation
                    strong_correlations.append(
                        (
                            self.feature_names[i],
                            self.feature_names[j],
                            corr_matrix[i][j],
                        )
                    )

        print("\nStrong Feature Correlations:")
        print("-" * 50)
        for feat1, feat2, corr in sorted(
            strong_correlations, key=lambda x: abs(x[2]), reverse=True
        ):
            print(f"{feat1} and {feat2}: {corr:.2f}")
