import numpy as np
from typing import List, Dict, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from generate_profiles2 import Profile
from basic_approach import calculate_age

class XGBoostRecommender:
    def __init__(self):
        self.model = None
        
    def _create_feature_vector(self, viewer_profile: Profile, candidate_profile: Profile) -> List[float]:
        """
        Convert both viewer and candidate profiles into a feature vector that captures their relationship
        """
        features = [
            # Candidate features
            candidate_profile.rent_budget[0],  # minimum budget
            candidate_profile.rent_budget[1],  # maximum budget
            calculate_age(candidate_profile.birth_date),
            int(candidate_profile.smoking == "Yes"),
            int(candidate_profile.activity_hours == "Night"),
            
            # Viewer features
            viewer_profile.rent_budget[0],
            viewer_profile.rent_budget[1],
            calculate_age(viewer_profile.birth_date),
            int(viewer_profile.smoking == "Yes"),
            int(viewer_profile.activity_hours == "Night"),
            
            # Relationship features
            abs(calculate_age(viewer_profile.birth_date) - calculate_age(candidate_profile.birth_date)),  # age difference
            abs(viewer_profile.rent_budget[0] - candidate_profile.rent_budget[0]),  # budget difference
            int(viewer_profile.smoking == candidate_profile.smoking),  # same smoking preference
            int(viewer_profile.activity_hours == candidate_profile.activity_hours),  # same activity hours
        ]
        return features
        
    def train(self, global_swipe_history: List[Tuple[Profile, Profile, bool]]):
        """
        Train model on global swipe history
        Args:
            global_swipe_history: List of (viewer_profile, candidate_profile, liked) tuples
        """
        # Convert profiles to feature vectors including both viewer and candidate info
        X = [self._create_feature_vector(viewer, candidate) 
             for viewer, candidate, _ in global_swipe_history]
        y = [1 if liked else 0 for _, _, liked in global_swipe_history]
        
        X_train, X_test, y_train, y_test = train_test_split(
            np.array(X), np.array(y), 
            test_size=0.2, 
            random_state=42
        )
        
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric=['logloss', 'auc'],
            learning_rate=0.1
        )
        
        self.model.fit(X_train, y_train)
        
    def predict_probability(self, viewer_profile: Profile, candidate_profile: Profile) -> float:
        """
        Predict probability of viewer liking candidate
        Returns float between 0 and 1
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Convert profiles to feature vector
        features = self._create_feature_vector(viewer_profile, candidate_profile)
        # Get probability of like (second column is probability of class 1)
        return self.model.predict_proba([features])[0][1]
        
    def recommend_profiles(self, viewer_profile: Profile, candidate_profiles: List[Profile], top_k: int = 10) -> List[Profile]:
        """
        Recommend top-k profiles for viewer using XGBoost predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Calculate probability scores for all candidates
        scores = [(candidate, self.predict_probability(viewer_profile, candidate))
                  for candidate in candidate_profiles]
        
        # Sort by probability score and return top-k profiles
        sorted_candidates = sorted(scores, key=lambda x: x[1], reverse=True)
        return [profile for profile, _ in sorted_candidates[:top_k]]