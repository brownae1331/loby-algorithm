from typing import List, Tuple
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from collections import defaultdict
from statistics import mean, stdev
from typing import Dict, List, Tuple, Optional

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
        
    def create_feature_vector(self, profile: Profile) -> List[float]:
        """
        Convert profile into feature vector for model input
        """
        features = [
            profile.rent_budget[0],  # minimum budget
            profile.rent_budget[1],  # maximum budget
            calculate_age(profile.birth_date),
            int(profile.smoking == "Yes"),
            int(profile.activity_hours == "Night"),
        ]
        return features
        
    def train(self, swipe_history: List[Tuple[Profile, bool]]):
        """
        Train model based on swipe history
        Args:
            swipe_history: List of (profile, liked) tuples
        """
        if len(swipe_history) < 10:  # Minimum swipes needed for training
            return
            
        # Convert profiles to feature vectors
        X = [self.create_feature_vector(profile) for profile, _ in swipe_history]
        y = [1 if liked else 0 for _, liked in swipe_history]
        
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
        
    def predict_probability(self, profile: Profile) -> float:
        """
        Predict the probability that the user will like a given profile
        Returns: Float between 0 and 1 (probability of like)
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
            
        # Convert profile to feature vector
        features = self.create_feature_vector(profile)
        
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

