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
from typing import List, Tuple, Optional, Dict
import xgboost as xgb
from sklearn.model_selection import train_test_split
from XGboost.generate_profiles2 import Profile
from app.rules_based import helper_functions as help_func
from xgboost_helper_functions import FeatureEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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
            "age_difference",
            "origin_country_match",
            "course_match",
            "university_match",
            "occupation_match",
            "industry_match",
            "smoking_match",
            "activity_hours_match",
            "gender_match",
            "location_score",
        ]

    def normalize_encoded(self, value: Optional[int], max_val: int) -> float:
        """Normalize encoded values to 0-1 range"""
        if value is None:
            return 0.0
        return value / max_val

    def create_feature_vector(
        self, viewer_profile: Profile, swiped_profile: Profile, loc_score: Optional[float] = None
    ) -> List[float]:
        """Create feature vector using standardized encodings from FeatureEncoder"""

        # Initialize features list
        features = [
            # Budget overlap
            help_func.CalculateScoreFunctions.calculate_budget_overlap_score(
                viewer_profile.rent_budget_range, swiped_profile.rent_budget_range
            ),
            # Age difference
            abs(
                help_func.calculate_age(viewer_profile.birth_date)
                - help_func.calculate_age(swiped_profile.birth_date)
            ),
            # Origin Country match
            1.0
            if viewer_profile.origin_country == swiped_profile.origin_country
            else 0.0,
            # Course match - context-aware
            self.course_match(viewer_profile, swiped_profile),
            # University match - context-aware
            self.university_match(viewer_profile, swiped_profile),
            # Occupation match
            1.0 if viewer_profile.occupation == swiped_profile.occupation else 0.0,
            # Industry match - context-aware
            self.industry_match(viewer_profile, swiped_profile),
            # Smoking match
            1.0 if viewer_profile.smoking == swiped_profile.smoking else 0.0,
            # Activity hours match
            1.0
            if viewer_profile.activity_hours == swiped_profile.activity_hours
            else 0.0,
            # Gender match
            1.0 if viewer_profile.gender == swiped_profile.gender else 0.0,
            # Location score (normalize to 0-1 range)
            float(loc_score)/0.2 if loc_score is not None else 0.5,  # Default to 0.5 when missing
        ]

        return features

    def university_match(self, viewer_profile: Profile, swiped_profile: Profile) -> float:
        """
        University matching with null handling:
        - If both have values and they match: 1.0
        - If both have values but don't match: 0.0
        - If either or both are null: 0.5 (neutral)
        """
        if viewer_profile.university_id and swiped_profile.university_id:
            return 1.0 if viewer_profile.university_id == swiped_profile.university_id else 0.0
        return 0.5  # Neutral score when either is null
    
    def course_match(self, viewer_profile: Profile, swiped_profile: Profile) -> float:
        """
        Course matching with null handling:
        - If both have values and they match: 1.0
        - If both have values but don't match: 0.0
        - If either or both are null: 0.5 (neutral)
        """
        if viewer_profile.course_id and swiped_profile.course_id:
            return 1.0 if viewer_profile.course_id == swiped_profile.course_id else 0.0
        return 0.5  # Neutral score when either is null
    
    def industry_match(self, viewer_profile: Profile, swiped_profile: Profile) -> float:
        """
        Industry matching with null handling:
        - If both have values and they match: 1.0
        - If both have values but don't match: 0.0
        - If either or both are null: 0.5 (neutral)
        """
        if viewer_profile.work_industry and swiped_profile.work_industry:
            return 1.0 if viewer_profile.work_industry == swiped_profile.work_industry else 0.0
        return 0.5  # Neutral score when either is null

    def train(self, global_swipe_history: List[Tuple[Profile, Profile, bool, Optional[float]]]):
        """
        Train model on global swipe history.
        Args:
            global_swipe_history: List of (viewer_profile, candidate_profile, liked_bool, location_score) tuples
        """
        # Convert profiles to feature vectors including both viewer and candidate info
        x = [
            self.create_feature_vector(viewer, candidate, loc_score)
            for (viewer, candidate, _, loc_score) in global_swipe_history
        ]
        y = [1 if liked else 0 for (_, _, liked, _) in global_swipe_history]

        x_train, x_test, y_train, y_test = train_test_split(
            np.array(x), np.array(y), test_size=0.2, random_state=42
        )

        self.model.fit(x_train, y_train)

        # Predict on the test set
        y_pred = self.model.predict(x_test)
        y_pred_proba = self.model.predict_proba(x_test)[:, 1]
        auc_score = roc_auc_score(y_test, y_pred_proba)

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
        print(f"AUC Score: {auc_score:.4f}")

    def predict_probability(
        self, viewer_profile: Profile, swiped_profile: Profile
    ) -> float:
        """
        Predict probability of a match between viewer and candidate.
        Args:
            viewer_profile: Profile of the viewer
            swiped_profile: Profile of the candidate
        Returns:
            Probability of a match
        """
        features = [self.create_feature_vector(viewer_profile, swiped_profile)]
        return self.model.predict_proba(features)[0, 1]

    def recommend_profiles(
        self, viewer_profile: Profile, swiped_profiles: List[Profile], 
        location_scores: Dict = None, top_k: int = 5
    ) -> List[Tuple[Profile, float]]:
        """
        Recommend profiles for a given viewer profile using hard filters first,
        then XGBoost model for ranking.
        
        Args:
            viewer_profile: The profile of the user viewing recommendations
            swiped_profiles: List of candidate profiles to rank
            location_scores: Optional dictionary mapping (user1_id, user2_id) to location scores
            top_k: Number of top recommendations to return
            
        Returns:
            List of (profile, score) tuples for top recommendations
        """
        # Initialize location_scores if not provided
        if location_scores is None:
            location_scores = {}
        

        # Apply hard filters first
        filtered_profiles = []
        
        # Track filter failures for reporting
        failed_date = failed_budget = failed_gender = failed_age = 0
        
        # Helper function to parse dates consistently
        def parse_date(date_val):
            if isinstance(date_val, str) and date_val.upper() == "IMMEDIATELY":
                return pd.Timestamp.now()
            try:
                return pd.to_datetime(date_val)
            except:
                return None
        
        # Parse viewer's available date
        viewer_available_at = parse_date(viewer_profile.available_at)
        
        for profile in swiped_profiles:
            # Skip if it's the same user
            if profile.user_id == viewer_profile.user_id:
                continue
            
            # 1. Date range check (within 2 weeks)
            if viewer_available_at and profile.available_at:
                profile_available_at = parse_date(profile.available_at)
                if profile_available_at:
                    if not (
                        (viewer_available_at - pd.Timedelta(days=14))
                        <= profile_available_at
                        <= (viewer_available_at + pd.Timedelta(days=14))
                    ):
                        failed_date += 1
                        continue
            
            # 2. Budget overlap check
            if viewer_profile.rent_budget_range and profile.rent_budget_range:
                viewer_min, viewer_max = viewer_profile.rent_budget_range
                candidate_min, candidate_max = profile.rent_budget_range
                
                # Check if there's no overlap in budget ranges
                if viewer_max < candidate_min or candidate_max < viewer_min:
                    failed_budget += 1
                    continue
            
            # 3. Age range check
            if hasattr(viewer_profile, 'age_range') and viewer_profile.age_range:
                min_age, max_age = viewer_profile.age_range
                candidate_age = help_func.calculate_age(profile.birth_date)
                
                if candidate_age < min_age or candidate_age > max_age:
                    failed_age += 1
                    continue
            
            # 4. Gender preference check
            if (viewer_profile.preferred_gender and profile.gender and 
                profile.preferred_gender and viewer_profile.gender):
                
                # Check if viewer's preference matches candidate's gender
                viewer_accepts_candidate = (
                    viewer_profile.preferred_gender == "ANY" or 
                    viewer_profile.preferred_gender == profile.gender
                )
                
                # Check if candidate's preference matches viewer's gender
                candidate_accepts_viewer = (
                    profile.preferred_gender == "ANY" or 
                    profile.preferred_gender == viewer_profile.gender
                )
                
                # Only continue if both accept each other
                if not (viewer_accepts_candidate and candidate_accepts_viewer):
                    failed_gender += 1
                    continue
            
            # If passed all hard filters, add to filtered profiles
            filtered_profiles.append(profile)
        
        # Print filter statistics
        print("\nFilter results:")
        print(f"Failed date range: {failed_date}")
        print(f"Failed budget overlap: {failed_budget}")
        print(f"Failed age preference: {failed_age}")
        print(f"Failed gender preference: {failed_gender}")
        print(f"Passed all filters: {len(filtered_profiles)}")
        
        # If no profiles passed the hard filters, return empty list
        if not filtered_profiles:
            return []
        
        # Create feature vectors for each candidate profile
        feature_vectors = []
        for candidate_profile in filtered_profiles:
            # Get location score for this pair if available
            loc_score = location_scores.get((viewer_profile.user_id, candidate_profile.user_id), None)
            
            feature_vector = self.create_feature_vector(
                viewer_profile, candidate_profile, loc_score
            )
            feature_vectors.append(feature_vector)
        
        # Convert to numpy array
        X = np.array(feature_vectors)
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X)[:, 1]  # Probability of class 1 (like)
        
        # Create (profile, probability) pairs and sort by probability
        recommendations = list(zip(filtered_profiles, probabilities))
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        # Print recommendations
        print("\nTop Recommendations:")
        print("-" * 60)
        for i, (profile, score) in enumerate(recommendations[:top_k], 1):
            print(f"\nRecommendation #{i} (Match Score: {score:.4f})")
            print(f"Profile ID: {profile.user_id}")
            print(f"Age: {help_func.calculate_age(profile.birth_date)}")
            print(f"Gender: {profile.gender}")
            print(f"Work Industry: {profile.work_industry or 'Not specified'}")
            print(f"University ID: {profile.university_id or 'Not specified'}")
            print(f"Course ID: {profile.course_id or 'Not specified'}")
            print(f"Activity Hours: {profile.activity_hours}")
            print(f"Smoking: {profile.smoking}")
            if profile.rent_budget_range:
                print(f"Budget Range: {profile.rent_budget_range[0]}-{profile.rent_budget_range[1]}")
            else:
                print("Budget Range: Not specified")
            print(f"Available From: {profile.available_at or 'Not specified'}")
            print("-" * 60)
        
        # Return top-k recommendations
        return recommendations[:top_k]

    def get_booster(self):
        if not hasattr(self.model, "get_booster"):
            raise ValueError("Model not trained yet!")
        return self.model.get_booster()

    def save_model(self, file_path):
        self.model.save_model(file_path)


if __name__ == "__main__":
    print("This file contains the XGBoostRecommender class.")
    print("To train and use the model, run train_from_csv.py instead.")
