from manage2 import XGBoostRecommender, UserPreferenceLearner
from generate_profiles2 import Profile
import pandas as pd
from datetime import datetime

def create_test_profile(user_id: int, budget: tuple, age: int, smoking: str, activity: str) -> Profile:
    """Helper function to create test profiles quickly"""
    return Profile(
        user_id=user_id,
        first_name=f"User{user_id}",
        last_name="Test",
        birth_date=pd.to_datetime(f"{2024-age}-01-01"),
        rent_budget=budget,
        smoking=smoking,
        activity_hours=activity,
        # ... other required fields with default values
    )

def test_recommender():
    # 1. Create some test profiles
    test_profiles = [
        create_test_profile(1, (500, 700), 25, "No", "Night"),
        create_test_profile(2, (800, 1000), 22, "Yes", "Night"),
        create_test_profile(3, (600, 800), 24, "No", "Morning"),
        create_test_profile(4, (900, 1200), 28, "Yes", "Morning"),
        create_test_profile(5, (500, 600), 23, "No", "Night"),
    ]
    
    # 2. Create swipe history (simulating user preferences)
    swipe_history = [
        (test_profiles[0], True),   # Likes: cheaper, non-smoker, night owl
        (test_profiles[1], False),  # Dislikes: expensive, smoker
        (test_profiles[2], True),   # Likes: mid-range, non-smoker
        (test_profiles[3], False),  # Dislikes: expensive, smoker
        (test_profiles[4], True),   # Likes: cheaper, non-smoker, night owl
    ]
    
    # 3. Initialize and train recommender
    recommender = XGBoostRecommender()
    recommender.train(swipe_history)
    
    # 4. Create new profiles to test recommendations
    candidate_profiles = [
        create_test_profile(6, (550, 750), 24, "No", "Night"),    # Should be high match
        create_test_profile(7, (1000, 1200), 29, "Yes", "Morning"), # Should be low match
        create_test_profile(8, (600, 800), 25, "No", "Night"),    # Should be high match
    ]
    
    # 5. Get recommendations
    recommendations = recommender.recommend_profiles(candidate_profiles)
    
    # 6. Print results
    print("\nRecommendations:")
    for profile, probability in recommendations:
        print(f"Profile {profile.user_id}:")
        print(f"  Budget: £{profile.rent_budget[0]}-{profile.rent_budget[1]}")
        print(f"  Age: {calculate_age(profile.birth_date)}")
        print(f"  Smoking: {profile.smoking}")
        print(f"  Activity: {profile.activity_hours}")
        print(f"  Match Probability: {probability*100:.1f}%\n")

if __name__ == "__main__":
    test_recommender() 