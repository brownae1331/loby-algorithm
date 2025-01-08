from app.rules_based.helper_functions import calculate_age, initialize_profile_list
from manage2 import XGBoostRecommender
from generate_profiles2 import Profile
import pandas as pd
from datetime import datetime


def create_test_profile(user_id: int, budget: tuple, age: int, smoking: str, activity: str) -> Profile:
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
    test_profiles = [
        create_test_profile(user_id=1, budget=(500, 700), age=25, smoking="No", activity="Night"),
        create_test_profile(user_id=2, budget=(800, 1000), age=22, smoking="Yes", activity="Night"),
        create_test_profile(user_id=3, budget=(600, 800), age=24, smoking="No", activity="Morning"),
        create_test_profile(user_id=4, budget=(900, 1200), age=28, smoking="Yes", activity="Morning"),
        create_test_profile(user_id=5, budget=(500, 600), age=23, smoking="No", activity="Night"),
    ]

    # 2. Create swipe history (simulating user preferences).
    #    The tuple is (viewer_profile, candidate_profile, liked_bool).
    swipe_history = [
        (test_profiles[0], test_profiles[1], True),  # user_id=1 likes user_id=2
        (test_profiles[1], test_profiles[3], False),  # user_id=2 dislikes user_id=4
        (test_profiles[2], test_profiles[0], True),  # user_id=3 likes user_id=1
        (test_profiles[3], test_profiles[2], False),  # user_id=4 dislikes user_id=3
        (test_profiles[4], test_profiles[4], True),  # user_id=5 "likes" themselves, just a filler example
    ]

    # 3. Initialize and train recommender
    recommender = XGBoostRecommender()
    recommender.train(swipe_history)

    # 4. Create a list of profiles we want to rank for a given viewer
    swiped_profiles = initialize_profile_list()  # e.g. from DB or a small dummy set

    # 5. Get top-k recommendations (now returns list of (Profile, probability) tuples)
    recommendations = recommender.recommend_profiles(viewer_profile=test_profiles[0],
                                                     swiped_profiles=swiped_profiles,
                                                     top_k=10)

    # 6. Print results
    print("\nRecommendations for user_id=1:")
    for profile, probability in recommendations:
        print(f"Profile {profile.user_id}:")
        print(f"  Budget: £{profile.rent_budget[0]}-{profile.rent_budget[1]}")
        print(f"  Age: {calculate_age(profile.birth_date)}")
        print(f"  Smoking: {profile.smoking}")
        print(f"  Activity: {profile.activity_hours}")
        print(f"  Match Probability: {probability * 100:.1f}%\n")


if __name__ == "__main__":
    test_recommender()
