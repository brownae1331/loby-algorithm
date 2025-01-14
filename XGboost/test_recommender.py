from app.rules_based.helper_functions import calculate_age, initialize_profile_list
from manage2 import XGBoostRecommender
from generate_profiles2 import Profile
import pandas as pd
from datetime import datetime
import xgboost as xgb

from app.rules_based.helper_functions import initialize_profile_list, calculate_overall_score
from generate_profiles2 import Profile
import matplotlib.pyplot as plt


def generate_swipe_history(profiles):
    swipe_history = []
    total_profiles = len(profiles)
    for i, viewer_profile in enumerate(profiles):
        for j, swiped_profile in enumerate(profiles):
            if i != j:
                score = calculate_overall_score(viewer_profile, swiped_profile)
                liked = score >= 0.8  # Assuming a threshold of 0.8 for "like"
                swipe_history.append((viewer_profile, swiped_profile, liked))
            if (j + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total_profiles} viewer profiles")
    return swipe_history



def create_test_profile(user_id: int, budget: tuple[int, int], age: int, smoking: str, activity: str) -> Profile:
    """
    Helper function to create test profiles quickly.
    Uses the passed arguments to set various fields.
    """
    # Approximate a birth_date given the target age
    # For example, if we want someone who is "25", we pick a date ~25 years in the past
    birth_year = datetime.now().year - age
    birth_date = datetime(birth_year, 1, 1)  # just fix month/day to Jan 1

    return Profile(
        user_id=user_id,
        first_name="Test",
        last_name="User",
        birth_date=pd.to_datetime(birth_date),
        is_verified=True,
        gender="Male",
        description="Testing user profile",
        languages=["English"],
        origin_country="UK",
        occupation="Student",
        work_industry=None,
        university_id="UCL",
        sexual_orientation="Heterosexual",
        pets="Dog",
        activity_hours=activity,
        smoking=smoking,
        extrovert_level=5,
        cleanliness_level=5,
        partying_level=5,
        sex_living_preference="Both",
        rent_location_preference="London",
        age_preference=(18, 35),
        rent_budget=budget,
        last_filter_processed_at=pd.to_datetime("2023-06-01"),
        available_at="2024-09",
        roommate_count_preference=2,
        interests=["Reading", "Traveling", "Cooking"],
        course="Computer Science"
    )


def test_recommender():
    # 1. Create some test profiles
    test_profiles = initialize_profile_list()

    # 2. Create swipe history (simulating user preferences).
    #    The tuple is (viewer_profile, candidate_profile, liked_bool).
    swipe_history = generate_swipe_history(test_profiles)

    # 3. Initialize and train recommender
    recommender = XGBoostRecommender()
    recommender.train(swipe_history)

    # 4. Create a list of profiles we want to rank for a given viewer
    swiped_profiles = test_profiles  # e.g. from DB or a small dummy set

    # 5. Get top-k recommendations (now returns list of (Profile, probability) tuples)
    recommendations = recommender.recommend_profiles(viewer_profile=test_profiles[0],
                                                     swiped_profiles=swiped_profiles,
                                                     top_k=50)

    profile1 = test_profiles[0]

    # 6. Print results
    print("\nRecommendations for user_id=1:")
    print(f"Profile {profile1.user_id}:")
    print(f"  Budget: £{profile1.rent_budget[0]}-{profile1.rent_budget[1]}")
    print(f"  Age: {calculate_age(profile1.birth_date)}")
    print(f"  Smoking: {profile1.smoking}")
    print(f"  Activity: {profile1.activity_hours}")

    print("\nRecommendations for all specified features:")
    for profile, probability in recommendations:
        print(f"Profile {profile.user_id}:")
        print(f"  Budget: £{profile.rent_budget[0]}-{profile.rent_budget[1]}")
        print(f"  Birth date: {calculate_age(profile.birth_date)}")
        print(f"  Origin country: {profile.origin_country}")
        print(f"  Course: {profile.course}")
        print(f"  Occupation: {profile.occupation}")
        print(f"  Work industry: {profile.work_industry}")
        print(f"  Smoking: {profile.smoking}")
        print(f"  University: {profile.university_id}")
        print(f"  Match Probability: {probability * 100:.1f}%\n")


    booster = recommender.get_booster()
    feature_importances = booster.get_score(importance_type='weight')

    feature_names = [
        "budget",
        "birth date",
        "origin country",
        "course",
        "occupation",
        "work industry",
        "smoking",
        "activity hours",
        "university"
    ]

    # Print feature importances
    print("Feature Importances:")
    for feature, importance in zip(feature_names, feature_importances.values()):
        print(f"{feature}: {importance}")

    recommender.save_model(r"C:\Users\jeetu\Desktop\loby-algorithm\XGboost")


if __name__ == "__main__":
    test_recommender()
