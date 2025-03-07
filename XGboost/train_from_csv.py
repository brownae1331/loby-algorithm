import sys
import os
import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime
import matplotlib.pyplot as plt

# Get absolute path to project root and add to Python path
project_root = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
app_path = os.path.join(project_root, "app")

# Add both paths
sys.path.append(project_root)
sys.path.append(app_path)

print(f"Project root: {project_root}")  # Debug print
print(f"App path: {app_path}")  # Debug print
print(f"Python path: {sys.path}")  # Debug print

# Import the Profile class from generate_profiles2.py instead of app/rules_based/generate_profiles.py
from XGboost.generate_profiles2 import Profile
from app.rules_based import helper_functions as help_func
# Import the XGBoostRecommender from manage2.py
from manage2 import XGBoostRecommender

def parse_range(range_str):
    """Parse a range string like '[400,801)' into a tuple (400, 801)"""
    if not range_str or pd.isna(range_str):
        return None
    # Remove brackets and parse numbers
    clean_str = str(range_str).replace('[', '').replace(')', '').replace(']', '')
    try:
        low, high = clean_str.split(',')
        return (int(float(low)), int(float(high)))
    except (ValueError, TypeError):
        print(f"Warning: Could not parse range string: {range_str}")
        return None


def load_profiles_from_csv(csv_path):
    """Load profiles from CSV file"""
    df = pd.read_csv(csv_path)
    profiles = {}

    for _, row in df.iterrows():
        try:
            profile = Profile(
                id=row["id"],
                user_id=row["user_id"],
                first_name=row.get("first_name", ""),
                last_name=row.get("last_name", ""),
                birth_date=pd.to_datetime(row["birth_date"]),
                is_verified=row.get("is_verified", False),
                gender=row.get("gender", ""),
                languages=row.get("languages", "").split(",")
                if pd.notna(row.get("languages"))
                else [],
                origin_country=row.get("origin_country", ""),
                occupation=row.get("occupation", ""),
                sexual_orientation=row.get("sexual_orientation", ""),
                pets=row.get("pets") if pd.notna(row.get("pets")) else None,
                activity_hours=row.get("activity_hours", ""),
                smoking=row.get("smoking", ""),
                extrovert_level=row.get("extrovert_level", 0),
                cleanliness_level=row.get("cleanliness_level", 0),
                partying_level=row.get("partying_level", 0),
                work_industry=row.get("work_industry")
                if pd.notna(row.get("work_industry"))
                else None,
                university_id=row.get("university_id")
                if pd.notna(row.get("university_id"))
                else None,
                course_id=row.get("course_id")
                if pd.notna(row.get("course_id"))
                else None,  
                created_at=row.get("created_at", None),
                contract_length=row.get("contract_length", None),
                age_range=parse_range(row.get("age_range", None)),
                preferred_gender =row.get("preferred_gender", None),
                rent_budget_range=parse_range(row.get("rent_budget_range", None)),
                available_at=row.get("available_at", None),
            )
            profiles[row["user_id"]] = profile
        except Exception as e:
            print(f"Error creating profile for row: {row}")
            print(f"Error: {str(e)}")
            continue

    return profiles


def generate_training_data(
    profiles: Dict[int, Profile], 
    likes_df: pd.DataFrame,  # positive interactions (likes)
    swipes_df: pd.DataFrame  # negative interactions (left swipes)
):
    """
    Generate training data using actual positive (likes) and negative (left swipes) interactions
    """
    training_data = []

    # Add positive samples from likes
    for _, row in likes_df.iterrows():
        viewer_id = row["profile_id_1"]
        liked_id = row["profile_id_2"]

        if viewer_id in profiles and liked_id in profiles:
            training_data.append(
                (
                    profiles[viewer_id],
                    profiles[liked_id],
                    True,  # positive case
                )
            )

    # Add negative samples from swipes
    for _, row in swipes_df.iterrows():
        viewer_id = row["profile_1_id"] 
        swiped_id = row["profile_2_id"]  

        if viewer_id in profiles and swiped_id in profiles:
            training_data.append(
                (
                    profiles[viewer_id],
                    profiles[swiped_id],
                    False,  # negative case
                )
            )

    print(f"Total training samples: {len(training_data)}")
    print(f"Positive samples: {sum(1 for _, _, label in training_data if label)}")
    print(f"Negative samples: {sum(1 for _, _, label in training_data if not label)}")

    return training_data


def calculate_age(birth_date):
    """Calculate age from birth date"""
    if birth_date is None:
        return 0
    today = datetime.now()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def print_feature_importance_with_names(model, feature_names):
    """Print feature importance with proper feature names"""
    # Get feature importance
    importance = model.get_booster().get_score(importance_type="gain")
    
    # Create a mapping from fX to actual feature names
    feature_map = {f"f{i}": name for i, name in enumerate(feature_names)}
    
    # Create a dictionary with proper feature names
    named_importance = {feature_map.get(feat, feat): imp for feat, imp in importance.items()}
    
    # Convert to DataFrame and sort
    importance_df = pd.DataFrame({
        "feature": list(named_importance.keys()),
        "importance": list(named_importance.values())
    })
    importance_df = importance_df.sort_values("importance", ascending=False)
    
    # Print features by importance
    print("\nFeature Importance with Proper Names:")
    print("-----------------------------------")
    for i, (feature, imp) in enumerate(zip(importance_df["feature"], importance_df["importance"])):
        print(f"{i+1}. {feature}: {imp:.4f}")
    
    # Plot with proper names
    plt.figure(figsize=(12, 10))
    plt.barh(
        importance_df["feature"].values[:min(20, len(importance_df))][::-1],
        importance_df["importance"].values[:min(20, len(importance_df))][::-1]
    )
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")
    plt.tight_layout()
    plt.savefig("feature_importance_named.png")
    print("\nFeature importance plot with proper names saved to feature_importance_named.png")


def main():
    # Get current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct paths to CSV files
    profiles_path = os.path.join(current_dir, "profiles+filters.csv")
    likes_path = os.path.join(current_dir, "profile_like_05-03.csv")
    swipes_path = os.path.join(current_dir, "profile_swipes_05-03.csv")
    
    # 1. Load profiles from CSV
    test_profiles = load_profiles_from_csv(profiles_path)

    # 2. Create swipe history (simulating user preferences)
    print(f"\nGenerating training data from likes and swipes")
    swipe_history = generate_training_data(
        test_profiles, 
        pd.read_csv(likes_path), 
        pd.read_csv(swipes_path)
    )

    # 3. Initialize and train recommender
    print(f"\nTraining XGBoost model")
    recommender = XGBoostRecommender()
    recommender.train(swipe_history)

    # Print feature importance with proper names
    print_feature_importance_with_names(recommender.model, recommender.feature_names)

    # 4. Create a list of profiles we want to rank for a given viewer
    # Convert profiles dictionary to list once
    profile_list = list(test_profiles.values())

    # 5. Generate recommendations for the first profile
    user_id = 202  # The user ID you want
    viewer_profile = next((p for p in profile_list if p.user_id == user_id), None)

    if viewer_profile:
        print(f"\nGenerating recommendations for user_id={user_id}")
        recommendations = recommender.recommend_profiles(
            viewer_profile=viewer_profile,
            swiped_profiles=profile_list,
            top_k=10
        )
    else:
        print(f"User ID {user_id} not found in profiles")

    # 6. Print results
    print("\nTop 5 recommendations:")
    for i, (profile, score) in enumerate(recommendations[:5]):
        print(f"{i+1}. User {profile.user_id}: {score:.4f} confidence")
    
    # 7. Save the model
    model_path = os.path.join(current_dir, "xgboost_model.json")
    recommender.save_model(model_path)
    print(f"\nModel saved to {model_path}")


if __name__ == "__main__":
    main()
