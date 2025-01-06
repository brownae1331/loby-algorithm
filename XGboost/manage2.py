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
        
    def _create_feature_vector(self, profile: Profile) -> List[float]:
        """
        Convert a profile into a numerical feature vector for XGBoost
        """
        features = [
            profile.rent_budget[0],  # minimum budget
            profile.rent_budget[1],  # maximum budget
            calculate_age(profile.birth_date),
            int(profile.smoking == "Yes"),  # convert to 0/1
            int(profile.activity_hours == "Night"),  # convert to 0/1
            # We'll add more features later
        ]
        return features
        
    def train(self, swipe_history: List[Tuple[Profile, bool]]):
        """
        Train XGBoost model on user's swipe history
        swipe_history: List of (Profile, liked) pairs
        """
        # Convert profiles to feature vectors
        X = [self._create_feature_vector(profile) for profile, _ in swipe_history]
        # Get labels (liked or not)
        y = [1 if liked else 0 for _, liked in swipe_history]
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Initialize and train XGBoost model
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            learning_rate=0.1
        )
        
        self.model.fit(X, y)
        
    def predict_probability(self, profile: Profile) -> float:
        """
        Predict the probability that the user will like a given profile
        Returns: Float between 0 and 1 (probability of like)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Convert profile to feature vector
        features = self._create_feature_vector(profile)
        
        # Reshape for single prediction (XGBoost expects 2D array)
        X = np.array([features])
        
        # Get probability of like (second column is probability of class 1)
        return self.model.predict_proba(X)[0][1]
        
    def recommend_profiles(self, candidate_profiles: List[Profile], top_k: int = 10) -> List[Tuple[Profile, float]]:
        """
        Recommend top k profiles from a list of candidates
        Returns: List of (Profile, probability) tuples, sorted by probability
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Get predictions for all profiles
        predictions = [(profile, self.predict_probability(profile)) 
                      for profile in candidate_profiles]
        
        # Sort by probability and return top k
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_k]