import os
import pandas as pd
import numpy as np
from datetime import datetime
from generate_profiles2 import Profile
from manage2 import XGBoostRecommender
import random
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from model_analyzer import ModelAnalyzer
import sys
import pickle

# Get the absolute path to the parent directory (i have to do this for some reason, but it works)
current_dir = os.path.dirname(os.path.abspath(__file__))  # Gets XGBoost directory
parent_dir = os.path.dirname(current_dir)  # Gets Loby_Algo directory

# Add the parent directory to Python's path
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from app.rules_based.helper_functions import calculate_age
from manage2 import XGBoostRecommender
from generate_profiles2 import Profile
import pandas as pd
from datetime import datetime
import xgboost as xgb


def load_profiles_from_csv(csv_path):
    """Load profiles from CSV file"""
    df = pd.read_csv(csv_path)
    profiles = {}

    # Print available columns for debugging
    print("\nAvailable columns in CSV:")
    print(df.columns.tolist())

    for _, row in df.iterrows():
        try:
            profile = Profile(
                user_id=row["user_id"],
                first_name=row.get("first_name", ""),
                last_name=row.get("last_name", ""),
                birth_date=pd.to_datetime(row["birth_date"]),
                is_verified=row.get("is_verified", False),
                gender=row.get("gender", ""),
                description=None,
                languages=row.get("languages", "").split(",")
                if pd.notna(row.get("languages"))
                else [],
                origin_country=row.get("origin_country", ""),
                occupation=row.get("occupation", ""),
                work_industry=row.get("work_industry")
                if pd.notna(row.get("work_industry"))
                else None,
                university_id=row.get("university_id")
                if pd.notna(row.get("university_id"))
                else None,
                course=row.get("course_id")
                if pd.notna(row.get("course_id"))
                else None,  # Changed from 'course' to 'course_id'
                sexual_orientation=row.get("sexual_orientation", ""),
                pets=row.get("pets") if pd.notna(row.get("pets")) else None,
                activity_hours=row.get("activity_hours", ""),
                smoking=row.get("smoking", ""),
                extrovert_level=row.get("extrovert_level", 0),
                cleanliness_level=row.get("cleanliness_level", 0),
                partying_level=row.get("partying_level", 0),
                sex_living_preference=None,
                rent_location_preference=None,
                age_preference=None,
                rent_budget=None,
                last_filter_processed_at=None,
                available_at=None,
                roommate_count_preference=None,
                interests=[],
            )
            profiles[row["user_id"]] = profile
        except Exception as e:
            print(f"Error creating profile for row: {row}")
            print(f"Error: {str(e)}")
            continue

    return profiles


def generate_training_data(
    profiles: dict[int, Profile], likes_df: pd.DataFrame, negative_ratio: float = 1.0
):  # negaive_ratio is a parameter
    """
    Generate training data with positive cases from likes and negative cases from non-likes
    negative_ratio: number of negative samples per positive sample
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

    # Generate negative samples
    all_user_ids = list(profiles.keys())
    num_negative_samples = int(len(training_data) * negative_ratio)

    negative_pairs = set()
    positive_pairs = set(
        (row["profile_id_1"], row["profile_id_2"]) for _, row in likes_df.iterrows()
    )

    # Pre-calculate valid candidate pairs to improve performance
    attempts = 0
    max_attempts = num_negative_samples * 10  # Prevent infinite loops

    while len(negative_pairs) < num_negative_samples and attempts < max_attempts:
        viewer_id = random.choice(all_user_ids)
        candidate_id = random.choice(all_user_ids)

        if (
            viewer_id != candidate_id
            and (viewer_id, candidate_id) not in positive_pairs
        ):
            negative_pairs.add((viewer_id, candidate_id))
            training_data.append(
                (
                    profiles[viewer_id],
                    profiles[candidate_id],
                    False,  # negative case
                )
            )
        attempts += 1

    if attempts >= max_attempts:
        print(
            f"Warning: Could only generate {len(negative_pairs)} negative samples out of {num_negative_samples} requested"
        )

    return training_data


def print_feature_importance(model):
    """
    Print and plot the feature importance of the trained XGBoost model.
    """
    # Define feature names in the exact order as they appear in the feature vector
    feature_names = [
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

    # This 'try' block is cursor
    try:
        # Get feature importances using XGBoost's native method
        booster = model.get_booster()
        importances = [
            booster.get_score(importance_type="gain").get(f"f{i}", 0)
            for i in range(len(feature_names))
        ]
        if not any(importances):  # Check if we got any non-zero importances
            print("[DEBUG] Cannot retrieve feature importances")
            return

        # Normalize the importances
        total_importance = sum(importances) if sum(importances) > 0 else 1
        normalized_scores = [score / total_importance for score in importances]

        # Print the feature importances
        print("\nFeature Importance Scores:")
        print("-" * 60)
        print(f"{'Feature Name':<40} {'Importance':>10}")
        print("-" * 60)

        # Create sorted pairs of (feature_name, importance)
        importance_pairs = list(
            zip(feature_names[: len(importances)], normalized_scores)
        )
        importance_pairs.sort(key=lambda x: x[1], reverse=True)

        # Print each feature and its importance
        for name, importance in importance_pairs:
            print(f"{name:<40} {importance:>10.3f}")

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.bar(
            [x[0] for x in importance_pairs],
            [x[1] for x in importance_pairs],
            color="skyblue",
        )
        plt.xlabel("Features")
        plt.ylabel("Normalized Importance")
        plt.title("Feature Importance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

        # Additionally, using XGBoost's built-in plot_importance
        booster = model.get_booster()
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(
            booster,
            max_num_features=10,
            importance_type="gain",
            xlabel="Gain",
            title="Feature Importance (Gain)",
        )
        plt.show()

    except Exception as e:
        print(f"\n[ERROR] An exception occurred in print_feature_importance: {str(e)}")
        import traceback

        traceback.print_exc()

    print("\n[DEBUG] Completed print_feature_importance function.")


def main():
    # Construct paths to CSV files
    profiles_path = os.path.join(current_dir, "profile.csv")
    likes_path = os.path.join(current_dir, "profile_like.csv")

    test_profiles = load_profiles_from_csv(profiles_path)

    # 2. Create swipe history (simulating user preferences).
    #    The tuple is (viewer_profile, candidate_profile, liked_bool).
    swipe_history = generate_training_data(test_profiles, pd.read_csv(likes_path))

    # 3. Initialize and train recommender
    recommender = XGBoostRecommender()
    recommender.train(swipe_history)

    # 4. Create a list of profiles we want to rank for a given viewer
    # Convert profiles dictionary to list once
    profile_list = list(test_profiles.values())

    # Use the same list for both purposes
    recommendations = recommender.recommend_profiles(
        viewer_profile=profile_list[0], swiped_profiles=profile_list, top_k=50
    )

    profile1 = profile_list[0]

    # 6. Print results
    print("\nRecommendations for user_id=1:")
    print(f"Profile {profile1.user_id}:")
    print(f"  Birth date: {calculate_age(profile1.birth_date)}")
    print(f"  Origin country: {profile1.origin_country}")
    print(f"  Course: {profile1.course}")
    # print(f"  Budget: £{profile1.rent_budget[0]}-{profile1.rent_budget[1]}")
    print(f"  Age: {calculate_age(profile1.birth_date)}")
    print(f"  Smoking: {profile1.smoking}")
    print(f"  Activity: {profile1.activity_hours}")

    print("\nRecommendations for all specified features:")
    for profile, probability in recommendations:
        print(f"Profile {profile.user_id}:")
        # print(f"  Budget: £{profile.rent_budget[0]}-{profile.rent_budget[1]}")
        print(f"  Birth date: {calculate_age(profile.birth_date)}")
        print(f"  Origin country: {profile.origin_country}")
        print(f"  Course: {profile.course}")
        print(f"  Occupation: {profile.occupation}")
        print(f"  Work industry: {profile.work_industry}")
        print(f"  Smoking: {profile.smoking}")
        print(f"  University: {profile.university_id}")
        print(f"  Match Probability: {probability * 100:.1f}%\n")

    booster = recommender.get_booster()
    feature_importances = booster.get_score(importance_type="weight")

    feature_names = [
        "budget",
        "birth date",
        "origin country",
        "course",
        "occupation",
        "work industry",
        "smoking",
        "activity hours",
        "university",
    ]

    # Print feature importances
    print("Feature Importances:")
    for feature, importance in zip(feature_names, feature_importances.values()):
        print(f"{feature}: {importance}")

    with open("recommender.pkl", "wb") as file:
        pickle.dump(recommender, file)


if __name__ == "__main__":
    main()
