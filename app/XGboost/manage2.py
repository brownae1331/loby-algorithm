import sys
import os
# Add the project root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from typing import List, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from generate_profiles2 import Profile
import rules_based.helper_functions as xgb_func


class XGBoostRecommender:
    def __init__(self):
        self.model = None

    def create_feature_vector(self, viewer_profile: Profile, swiped_profile: Profile) -> List[float]:
        """
        Convert both viewer and candidate profiles into a feature vector that captures their relationship
        """
        features = [
            # Compatibility scores
            xgb_func.CalculateScoreFunctions.calculate_budget_overlap_score(
            viewer_profile.rent_budget, swiped_profile.rent_budget),
            xgb_func.CalculateScoreFunctions.calculate_age_similarity_score(
                xgb_func.calculate_age(viewer_profile.birth_date), xgb_func.calculate_age(swiped_profile.birth_date)),
            xgb_func.ComparisonFunctions.compare_origin_country(
                viewer_profile.origin_country, swiped_profile.origin_country),
            xgb_func.ComparisonFunctions.compare_course(
                viewer_profile.course, swiped_profile.course),
            xgb_func.ComparisonFunctions.compare_occupation(
                viewer_profile.occupation, swiped_profile.occupation),
            xgb_func.ComparisonFunctions.compare_work_industry(
                viewer_profile.work_industry, swiped_profile.work_industry),
            xgb_func.ComparisonFunctions.compare_smoking(
                viewer_profile.smoking, swiped_profile.smoking),
            xgb_func.ComparisonFunctions.compare_activity_hours(
                viewer_profile.activity_hours, swiped_profile.activity_hours),
            xgb_func.ComparisonFunctions.compare_university(
                viewer_profile.university_id, swiped_profile.university_id)
        ]
        print(f"Feature vector for viewer {viewer_profile.user_id} and swiped {swiped_profile.user_id}: {features}")
        return features

    def train(self, global_swipe_history: List[Tuple[Profile, Profile, bool]]):
        """
        Train model on global swipe history.
        Args:
            global_swipe_history: List of (viewer_profile, candidate_profile, liked_bool) tuples
        """
        # Convert profiles to feature vectors including both viewer and candidate info
        x = [self.create_feature_vector(viewer, candidate)
             for viewer, candidate, _ in global_swipe_history]
        y = [1 if liked else 0 for _, _, liked in global_swipe_history]
        
        x_train, x_test, y_train, y_test = train_test_split(
            np.array(x),
            np.array(y),
            test_size=0.2,
            random_state=42
        )

        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'auc'],
            learning_rate=0.1
        )

        self.model.fit(x_train, y_train)
        
    def predict_probability(self, viewer_profile: Profile, swiped_profile: Profile) -> float:
        """
        Predict probability of viewer liking candidate.
        Returns a float between 0 and 1.
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        features = self.create_feature_vector(viewer_profile, swiped_profile)
        # Extract just the probability for class 1 (positive class)
        proba = self.model.predict_proba([features])[0][1]
        return float(proba)  # Ensure we return a float, not a numpy array or list
        
    def recommend_profiles(self, viewer_profile: Profile, swiped_profile: List[Profile], top_k: int = 10) -> List[Profile]:
        """
        Recommend top-k profiles for viewer using XGBoost predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")

        # Calculate probability scores for all candidates
        scores = [(candidate, self.predict_probability(viewer_profile, candidate))
                  for candidate in swiped_profile]
        
        # Sort by probability score and return top-k profiles
        sorted_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
        return [profile for profile, _ in sorted_candidates[:top_k]]