import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import xgboost as xgb
from sklearn.model_selection import train_test_split
from generate_profiles2 import Profile

class UserPreferenceLearner:
    def __init__(self, min_swipes: int = 20):
        self.min_swipes_for_personalization = min_swipes
        self.user_preferences: Dict[int, Dict] = {}
        
        # Initial weights from Rules based approach
        self.default_weights = {
            'budget_weight': 1.0,
            'age_similarity_weight': 1.0,
            'origin_country_weight': 1.0,
            'course_weight': 1.0,
            'occupation_weight': 1.0,
            'work_industry_weight': 1.0,
            'smoking_weight': 1.0,
            'activity_hours_weight': 1.0,
            'university_weight': 1.0
        }
        
    def learn_initial_preferences(self, user_id: int, swipe_history: List[Tuple[Profile, bool]]) -> Dict:
        """
        Learn initial user preferences from first few swipes
        Returns dict of feature weights
        """
        if len(swipe_history) < self.min_swipes_for_personalization:
            return self.default_weights
            
        weights = self.default_weights.copy()
        liked_profiles = [profile for profile, liked in swipe_history if liked]
        
        if liked_profiles:
            # Calculate average features of liked profiles
            avg_budget: float = np.mean([p.rent_budget[0] for p in liked_profiles])
            avg_age: float = np.mean([calculate_age(p.birth_date) for p in liked_profiles])
            
            # Adjust weights based on variance in liked profiles
            weights['budget_weight'] = self._calculate_feature_importance(
                [p.rent_budget[0] for p in liked_profiles]
            )
            weights['age_similarity_weight'] = self._calculate_feature_importance(
                [calculate_age(p.birth_date) for p in liked_profiles]
            )
            # ... adjust other weights similarly
            
        self.user_preferences[user_id] = weights
        return weights
        
    def _calculate_feature_importance(self, values: List[float]) -> float:
        """
        Calculate feature importance based on variance
        Low variance = High importance (user has strong preference)
        """
        if not values:
            return self.default_weights['budget_weight']
            
        variance = np.var(values)
        # Convert variance to weight: higher variance = lower weight
        weight = 1.0 / (1.0 + variance)
        return min(max(weight, 0.1), 2.0)  # Clamp between 0.1 and 2.0

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