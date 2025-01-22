import sys
import os

# Get absolute path to project root and add to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
app_path = os.path.join(project_root, 'app')

# Add both paths
sys.path.append(project_root)
sys.path.append(app_path)

print(f"Project root: {project_root}")  # Debug print
print(f"App path: {app_path}")          # Debug print
print(f"Python path: {sys.path}")       # Debug print

import numpy as np
from typing import List, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from generate_profiles2 import Profile
from app.rules_based import helper_functions as help_func
import xgboost_helper_functions as xgb_func


class XGBoostRecommender:
    def __init__(self):
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'auc'],
            learning_rate=0.1
        )

    def create_feature_vector(self, viewer_profile: Profile, swiped_profile: Profile) -> List[float]:
        """
        Convert both viewer and candidate profiles into a feature vector that captures their relationship
        """
        # Simple gender encoding function
        def encode_gender(gender: str) -> float:
            if not gender:
                return 0.0
            gender = gender.lower()
            if gender == 'male':
                return 1.0
            elif gender == 'female':
                return 2.0
            else:
                return 0.0  # for other/unknown
            
        features = [
            # Budget and age features
            help_func.CalculateScoreFunctions.calculate_budget_overlap_score(
                viewer_profile.rent_budget, swiped_profile.rent_budget
            ),
            help_func.calculate_age(viewer_profile.birth_date),  # viewer age
            help_func.calculate_age(swiped_profile.birth_date),  # candidate age
            abs(help_func.calculate_age(viewer_profile.birth_date) - 
                help_func.calculate_age(swiped_profile.birth_date)),  # age difference

            # Basic matching features
            help_func.ComparisonFunctions.compare_origin_country(
                viewer_profile.origin_country, swiped_profile.origin_country
            ),
            help_func.ComparisonFunctions.compare_course(
                viewer_profile.course, swiped_profile.course
            ),
            help_func.ComparisonFunctions.compare_occupation(
                viewer_profile.occupation, swiped_profile.occupation
            ),
            help_func.ComparisonFunctions.compare_work_industry(
                viewer_profile.work_industry, swiped_profile.work_industry
            ),
            help_func.ComparisonFunctions.compare_smoking(
                viewer_profile.smoking, swiped_profile.smoking
            ),
            help_func.ComparisonFunctions.compare_activity_hours(
                viewer_profile.activity_hours, swiped_profile.activity_hours
            ),
            help_func.ComparisonFunctions.compare_university(
                viewer_profile.university_id, swiped_profile.university_id
            ),

            # Personality and lifestyle features
            abs(viewer_profile.extrovert_level - swiped_profile.extrovert_level),  # extrovert difference
            abs(viewer_profile.cleanliness_level - swiped_profile.cleanliness_level),  # cleanliness difference
            abs(viewer_profile.partying_level - swiped_profile.partying_level),  # partying difference

            # Gender and orientation features
            encode_gender(viewer_profile.gender),  # viewer gender
            encode_gender(swiped_profile.gender),  # candidate gender
            1.0 if viewer_profile.gender == swiped_profile.gender else 0.0,  # gender match
            1.0 if viewer_profile.sexual_orientation == swiped_profile.sexual_orientation else 0.0,  # simple orientation match

            # Additional compatibility features
            1.0 if viewer_profile.pets == swiped_profile.pets else 0.0,  # simple pets match
            len(set(viewer_profile.languages) & set(swiped_profile.languages)) / max(
                len(viewer_profile.languages), len(swiped_profile.languages), 1
            )  # languages overlap
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
        x = [
            self.create_feature_vector(viewer, candidate)
            for (viewer, candidate, _) in global_swipe_history
        ]
        y = [1 if liked else 0 for (_, _, liked) in global_swipe_history]

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(x),
            np.array(y),
            test_size=0.2,
            random_state=42
        )

        self.model.fit(x_train, y_train)

    def predict_probability(self, viewer_profile: Profile, swiped_profile: Profile) -> float:
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
            self,
            viewer_profile: Profile,
            swiped_profiles: List[Profile],
            top_k: int = 100
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


