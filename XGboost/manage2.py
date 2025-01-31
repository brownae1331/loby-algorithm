import sys
import os

# Get absolute path to project root and add to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
app_path = os.path.join(project_root, "app")

# Add both paths
sys.path.append(project_root)
sys.path.append(app_path)

print(f"Project root: {project_root}")  # Debug print
print(f"App path: {app_path}")  # Debug print
print(f"Python path: {sys.path}")  # Debug print

import numpy as np
from typing import List, Tuple, Optional
import xgboost as xgb
from sklearn.model_selection import train_test_split
from app.rules_based.generate_profiles import Profile
from app.rules_based import helper_functions as help_func
from xgboost_helper_functions import FeatureEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
from model_analyzer import ModelAnalyzer


class XGBoostRecommender:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric=["logloss", "auc"],
            learning_rate=0.1,
        )
        self.feature_names = self.get_feature_names()
        self.encoder = FeatureEncoder()

    def get_feature_names(self) -> List[str]:
        return [
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
            "gender_match",
            "viewer_gender_code",
            "candidate_gender_code",
            "language_overlap",
            "viewer_language_count",
            "candidate_language_count",
            "has_english_match",
        ]

    def normalize_encoded(self, value: Optional[int], max_val: int) -> float:
        """Normalize encoded values to 0-1 range"""
        if value is None:
            return 0.0
        return value / max_val

    def create_feature_vector(
        self, viewer_profile: Profile, swiped_profile: Profile
    ) -> List[float]:
        """Create feature vector using standardized encodings from FeatureEncoder"""

        # Initialize features list
        features = [
            # Budget overlap
            help_func.CalculateScoreFunctions.calculate_budget_overlap_score(
                viewer_profile.rent_budget, swiped_profile.rent_budget
            ),
            # Age features
            help_func.calculate_age(viewer_profile.birth_date),
            help_func.calculate_age(swiped_profile.birth_date),
            abs(
                help_func.calculate_age(viewer_profile.birth_date)
                - help_func.calculate_age(swiped_profile.birth_date)
            ),
            # Origin Country features
            1.0
            if viewer_profile.origin_country == swiped_profile.origin_country
            else 0.0,
            self.normalize_encoded(
                self.encoder.get_origin_country_code(viewer_profile.origin_country), 196
            ),
            self.normalize_encoded(
                self.encoder.get_origin_country_code(swiped_profile.origin_country), 196
            ),
            # Course features
            1.0 if viewer_profile.course == swiped_profile.course else 0.0,
            self.normalize_encoded(
                self.encoder.get_course_code(viewer_profile.course), 250
            ),
            self.normalize_encoded(
                self.encoder.get_course_code(swiped_profile.course), 250
            ),
            # University features
            1.0
            if viewer_profile.university_id == swiped_profile.university_id
            else 0.0,
            self.normalize_encoded(
                self.encoder.get_university_id_code(viewer_profile.university_id), 250
            ),
            self.normalize_encoded(
                self.encoder.get_university_id_code(swiped_profile.university_id), 250
            ),
            # Occupation features
            1.0 if viewer_profile.occupation == swiped_profile.occupation else 0.0,
            self.normalize_encoded(
                self.encoder.get_occupation_code(viewer_profile.occupation), 3
            ),
            self.normalize_encoded(
                self.encoder.get_occupation_code(swiped_profile.occupation), 3
            ),
            # Work industry features
            1.0
            if viewer_profile.work_industry == swiped_profile.work_industry
            else 0.0,
            self.normalize_encoded(
                self.encoder.get_work_industry_code(viewer_profile.work_industry), 19
            ),
            self.normalize_encoded(
                self.encoder.get_work_industry_code(swiped_profile.work_industry), 19
            ),
            # Smoking features
            1.0 if viewer_profile.smoking == swiped_profile.smoking else 0.0,
            self.normalize_encoded(
                self.encoder.get_smoking_code(viewer_profile.smoking), 3
            ),
            self.normalize_encoded(
                self.encoder.get_smoking_code(swiped_profile.smoking), 3
            ),
            # Activity hours features
            1.0
            if viewer_profile.activity_hours == swiped_profile.activity_hours
            else 0.0,
            self.normalize_encoded(
                self.encoder.get_activity_hours_code(viewer_profile.activity_hours), 2
            ),
            self.normalize_encoded(
                self.encoder.get_activity_hours_code(swiped_profile.activity_hours), 2
            ),
            # Gender features
            1.0 if viewer_profile.gender == swiped_profile.gender else 0.0,
            self.normalize_encoded(
                self.encoder.get_gender_code(viewer_profile.gender), 2
            ),
            self.normalize_encoded(
                self.encoder.get_gender_code(swiped_profile.gender), 2
            ),
        ]

        # Handle language features separately
        viewer_languages = set(viewer_profile.languages or [])
        candidate_languages = set(swiped_profile.languages or [])

        # Add language features to the list
        features.extend(
            [
                # Original language overlap feature
                len(viewer_languages & candidate_languages)
                / max(len(viewer_languages), len(candidate_languages), 1),
                # Number of languages each person speaks (normalized to 0-1 range assuming max 5 languages)
                len(viewer_languages) / 5,
                len(candidate_languages) / 5,
                # Specific check for English as a common language
                1.0 if "English" in (viewer_languages & candidate_languages) else 0.0,
            ]
        )

        return features

    def train(self, global_swipe_history: List[Tuple[Profile, Profile, bool]]):
        """
        Train model on global swipe history.
        Args:
            global_swipe_history: List of (viewer_profile, candidate_profile, liked_bool) tuples
        """
        # Convert profiles to feature vectors including both viewer and candidate info
        x = [
            self.create_feature_vector(viewer, candidate)
            for (viewer, candidate, _) in global_swipe_history
        ]
        y = [1 if liked else 0 for (_, _, liked) in global_swipe_history]

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(x), np.array(y), test_size=0.2, random_state=42
        )

        self.model.fit(x_train, y_train)

        # Predict on the test set
        y_pred = self.model.predict(x_test)

        # Calculate performance metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Print performance metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

    def predict_probability(
        self, viewer_profile: Profile, swiped_profile: Profile
    ) -> float:
        """
        Predict probability of viewer liking candidate.
        Returns a float between 0 and 1.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Convert profiles to feature vector
        features = self.create_feature_vector(viewer_profile, swiped_profile)
        # Get probability of "liked" (index 1 is probability of class 1)

        return self.model.predict_proba([features])[0][1]

    def recommend_profiles(
        self, viewer_profile: Profile, swiped_profiles: List[Profile], top_k: int = 50
    ) -> List[Tuple[Profile, float]]:
        """
        Recommend top-k profiles for viewer using XGBoost predictions.
        Returns a list of (Profile, probability) sorted by descending probability.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Calculate probability scores for all candidates
        scores = [
            (candidate, self.predict_probability(viewer_profile, candidate))
            for candidate in swiped_profiles
        ]

        # Sort by probability score and return top-k profiles (with their scores)
        sorted_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
        return sorted_candidates[:top_k]

    def get_booster(self):
        if self.model is None:
            raise ValueError("Model not trained yet!")
        return self.model.get_booster()

    def save_model(self, file_path):
        self.model.save_model(file_path)

    # def analyze_feature_importance(self, plot: bool = True):
    #     """
    #     Analyze and visualize which features contribute most to matches
    #     """
    #     importance = self.model.get_booster().get_score(importance_type='gain')
    #     sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    #     print("\nFeature Importance Analysis:")
    #     print("-" * 50)
    #     for feature, score in sorted_importance:
    #         print(f"{feature}: {score:.4f}")

    #     if plot:
    #         plt.figure(figsize=(10, 6))
    #         features, scores = zip(*sorted_importance)
    #         plt.barh(features, scores)
    #         plt.title('Feature Importance')
    #         plt.xlabel('Importance Score')
    #         plt.tight_layout()
    #         plt.show()

    # def evaluate_model(self, X_test, y_test):
    #     """
    #     Comprehensive model evaluation with metrics and visualizations
    #     """
    #     y_pred = self.model.predict(X_test)
    #     y_prob = self.model.predict_proba(X_test)

    #     # Calculate metrics
    #     metrics = {
    #         'Accuracy': accuracy_score(y_test, y_pred),
    #         'Precision': precision_score(y_test, y_pred),
    #         'Recall': recall_score(y_test, y_pred),
    #         'F1 Score': f1_score(y_test, y_pred)
    #     }

    #     print("\nModel Performance Metrics:")
    #     print("-" * 50)
    #     for metric, value in metrics.items():
    #         print(f"{metric}: {value:.4f}")

    #     # Plot confusion matrix
    #     cm = confusion_matrix(y_test, y_pred)
    #     plt.figure(figsize=(8, 6))
    #     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    #     plt.title('Confusion Matrix')
    #     plt.ylabel('True Label')
    #     plt.xlabel('Predicted Label')
    #     plt.show()

    # def analyze_matching_patterns(self, X_test, y_test):
    #     """
    #     Analyze patterns in successful and unsuccessful matches
    #     """
    #     X_test_np = np.array(X_test)
    #     successful_matches = X_test_np[y_test == 1]
    #     unsuccessful_matches = X_test_np[y_test == 0]

    #     print("\nMatching Pattern Analysis:")
    #     print("-" * 50)

    #     # Analyze key features in successful matches
    #     print("\nSuccessful Matches Pattern:")
    #     self._analyze_feature_patterns(successful_matches)

    #     # Compare with unsuccessful matches
    #     print("\nUnsuccessful Matches Pattern:")
    #     self._analyze_feature_patterns(unsuccessful_matches)

    # def _analyze_feature_patterns(self, matches):
    #     """Helper method to analyze patterns in matches"""
    #     if len(matches) == 0:
    #         print("No matches to analyze")
    #         return

    #     # Calculate mean values for key features
    #     feature_means = np.mean(matches, axis=0)
    #     feature_stds = np.std(matches, axis=0)

    #     key_features = {
    #         'Budget Overlap': 0,
    #         'Age Difference': 3,
    #         'Origin Country Match': 4,
    #         'Course Match': 7,
    #         'Occupation Match': 13,
    #         'Smoking Match': 19,
    #         'Activity Hours Match': 22
    #     }

    #     for feature_name, idx in key_features.items():
    #         print(f"{feature_name}:")
    #         print(f"  Mean: {feature_means[idx]:.2f}")
    #         print(f"  Std: {feature_stds[idx]:.2f}")

    # def find_strong_correlations(self, X_test):
    #     """
    #     Find strong correlations between features
    #     """
    #     corr_matrix = np.corrcoef(np.array(X_test).T)
    #     strong_correlations = []

    #     for i in range(len(corr_matrix)):
    #         for j in range(i + 1, len(corr_matrix)):
    #             if abs(corr_matrix[i][j]) > 0.5:  # Threshold for strong correlation
    #                 strong_correlations.append((i, j, corr_matrix[i][j]))

    #     print("\nStrong Feature Correlations:")
    #     print("-" * 50)
    #     for i, j, corr in sorted(strong_correlations, key=lambda x: abs(x[2]), reverse=True):
    #         print(f"Feature {i} and Feature {j}: {corr:.2f}")


if __name__ == "__main__":
    recommender = XGBoostRecommender()
    # ... train the model ...

    # Create analyzer with your trained model
    analyzer = ModelAnalyzer(recommender.model)

    # Use the analyzer methods
    analyzer.analyze_feature_importance()
    analyzer.evaluate_model(X_test, y_test)  # Use your actual test data
    analyzer.analyze_matching_patterns(X_test, y_test)
    analyzer.find_strong_correlations(X_test)
