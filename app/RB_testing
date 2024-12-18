import pandas as pd
from typing import List, Tuple
from helper_functions import PrintFunctions, generate_likes, modify_weights_with_weighted_average, calculate_overall_score, calculate_age, assign_profiles_to_profile_list
from generate_profiles import Profile
import openpyxl
import random

# Alex's profile
starting_profile = Profile(
    user_id=0,
    first_name="Alex",
    last_name="Example",
    birth_date=pd.to_datetime("2002-05-22"),
    is_verified=True,
    gender="Male",
    description="Enjoys outdoor activities and reading",
    languages=["English", "Spanish"],
    origin_country="UK",
    occupation="Working",
    work_industry="Tech",
    university_id="UCL",
    course= None,
    sexual_orientation="Heterosexual",
    pets="Dog",
    activity_hours="Morning",
    smoking="No",
    extrovert_level=7,
    cleanliness_level=8,
    partying_level=4,
    sex_living_preference="Both",
    rent_location_preference="London",
    age_preference=(22, 30),
    rent_budget=(600, 900),
    last_filter_processed_at=pd.to_datetime("2023-06-01"),
    available_at="2024-09",
    roommate_count_preference=2,
    interests=["Reading", "Traveling", "Cooking"]
)

def create_test_profiles() -> List[Profile]:
    # Base template for all profiles
    base_profile = {
        "last_name": "User",
        "is_verified": True,
        "gender": "Male",
        "description": "Test profile",
        "languages": ["English"],
        "sexual_orientation": "Heterosexual",
        "pets": "None",
        "extrovert_level": 5,
        "cleanliness_level": 5,
        "partying_level": 5,
        "sex_living_preference": "Both",
        "rent_location_preference": "London",
        "age_preference": (22, 30),
        "last_filter_processed_at": pd.to_datetime("2023-06-01"),
        "available_at": "2024-09",
        "roommate_count_preference": 2,
        "interests": ["Reading", "Sports"]
    }
    
    # Specific variations for each profile
    profile_variations = [
        {
            "user_id": 396,
            "birth_date": "2004-01-01",
            "origin_country": "UK",
            "work_industry": "Finance",
            "course": "none",
            "smoking": "No",
            "university_id": "UCL",
            "rent_budget": (499, 760),
            "occupation": "Working",
            "activity_hours": "Morning"
        },
        {
            "user_id": 468,
            "birth_date": "2000-01-01",
            "origin_country": "UK",
            "work_industry": "Tech",
            "course": "none",
            "smoking": "Yes",
            "university_id": "QMUL",
            "rent_budget": (574, 661),
            "occupation": "Working",
            "activity_hours": "Morning"
        },
        {
            "user_id": 8,
            "birth_date": "2002-01-01",
            "origin_country": "USA",
            "work_industry": "Tech",
            "course": None,
            "smoking": None,
            "university_id": None,
            "rent_budget": (748, 949),
            "occupation": "Working",
            "activity_hours": "Night"
        },
        {
            "user_id": 112,
            "birth_date": "2002-01-01",  # 24 years old
            "origin_country": "Australia",
            "course": "Engineering",
            "occupation": "Finance",
            "work_industry": None,
            "smoking": "Yes",
            "university_id": "QMUL",
            "activity_hours": "Morning",
            "rent_budget": (737, 1603)
        },
        {
            "user_id": 384,
            "birth_date": "1998-01-01",  # 26 years old
            "origin_country": "Canada",
            "course": "none",
            "occupation": "Working",
            "work_industry": "Finance",
            "university_id": "UCL",
            "smoking": "No",
            "activity_hours": "Night",
            "rent_budget": (735, 1841)
        },
        {
            "user_id": 41,
            "birth_date": "2000-01-01",  # 24 years old
            "origin_country": "Australia",
            "course": None,
            "occupation": "Working",
            "work_industry": "Finance",
            "smoking": "Yes",
            "university_id": None,
            "activity_hours": "Morning",
            "rent_budget": (615, 1095)
        },
        {
            "user_id": 4,
            "birth_date": "2004-01-01",  # 26 years old
            "origin_country": "Australia",
            "course": None,
            "occupation": "Working",
            "work_industry": "Media",
            "smoking": "No",
            "university_id": None,
            "rent_budget": (634, 1646),
            "activity_hours": "Morning"
        },
        {
            "user_id": 102,
            "birth_date": "2000-01-01",  # 22 years old
            "origin_country": "USA",
            "course": None,
            "occupation": "Working",
            "work_industry": "Finance",
            "smoking": None,
            "university_id": "UCL",
            "activity_hours": "Morning",
            "rent_budget": (549, 673)
        },
    ]
    
    # Create profiles by combining base template with variations
    test_profiles = [
        Profile(
            first_name=f"Test{var['user_id']}",
            birth_date=pd.to_datetime(var['birth_date']),
            **{**base_profile, 
               **{k: (None if v == "None" else v) for k, v in var.items() if k not in ['birth_date']}}
        ) for var in profile_variations
    ]
    
    return test_profiles


def optimize_weights(starting_profile: Profile, profile_list: List[Profile], desired_order: List[int]) -> dict:
    """Optimize weights to match desired profile ordering"""
    default_weights = {
        'budget_weight': 0.059,
        'age_similarity_weight': 0.281,
        'origin_country_weight': 0.026,
        'course_weight': 0.187,
        'occupation_weight': 0.161,
        'work_industry_weight': 0.04,
        'smoking_weight': 0.036,
        'activity_hours_weight': 0.063
    }
    
    best_similarity = 0
    best_weights = default_weights.copy()
    
    print("Starting optimization...")
    print("Desired order:", desired_order)
    
    # Try different weight combinations
    for i in range(2000):
        # Generate weights between 0.1 and 10.0
        new_weights = {k: random.uniform(0.01, 0.3) for k in default_weights.keys()}
        
        # Update the weight attributes directly
        for p in [starting_profile] + profile_list:
            p.age_similarity_weight = new_weights['age_similarity_weight']
            p.occupation_weight = new_weights['occupation_weight']
            p.work_industry_weight = new_weights['work_industry_weight']
            p.activity_hours_weight = new_weights['activity_hours_weight']
            p.smoking_weight = new_weights['smoking_weight']
            p.budget_weight = new_weights['budget_weight']
            p.origin_country_weight = new_weights['origin_country_weight']
        
        # Calculate scores using the new weights
        scores = []
        for p in profile_list:
            score = calculate_overall_score(starting_profile, p)
            scores.append((p, score))
            
        current_order = [p.user_id for p, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
        
        # Debug print every 100 iterations
        if i % 100 == 0:
            print(f"\nIteration {i}")
            print("Current weights:", {k: round(v, 3) for k, v in new_weights.items()})
            print("Current order:", current_order)
            print("Current scores:", [(p.user_id, round(s, 3)) for p, s in sorted(scores, key=lambda x: x[1], reverse=True)])
        
        # Calculate similarity
        similarity = calculate_ordering_similarity(current_order, desired_order)
        
        if similarity > best_similarity:
            best_similarity = similarity
            best_weights = new_weights.copy()
            print(f"\nNew best similarity: {best_similarity:.3f}")
            print("Current order:", current_order)
            print("Desired order:", desired_order)
            print("Weights:", {k: round(v, 3) for k, v in new_weights.items()})
            
            # If we achieve perfect matching, we can stop
            if similarity >= 0.99:
                break
    
    # Update final weights
    for p in [starting_profile] + profile_list:
        p.age_similarity_weight = best_weights['age_similarity_weight']
        p.occupation_weight = best_weights['occupation_weight']
        p.work_industry_weight = best_weights['work_industry_weight']
        p.activity_hours_weight = best_weights['activity_hours_weight']
        p.smoking_weight = best_weights['smoking_weight']
        p.budget_weight = best_weights['budget_weight']
        p.origin_country_weight = best_weights['origin_country_weight']
    
    print("\nOptimization completed!")
    print(f"Best similarity achieved: {best_similarity:.3f}")
    print("Final weights:", {k: round(v, 3) for k, v in best_weights.items()})
    
    return best_weights

def calculate_ordering_similarity(current_order: List[int], desired_order: List[int]) -> float:
    """
    Calculate similarity between two orderings using weighted position differences.
    Places more importance on matching the top positions.
    """
    if len(current_order) != len(desired_order):
        return 0.0
    
    total_positions = len(desired_order)
    weighted_difference = 0
    
    for i, profile_id in enumerate(current_order):
        if profile_id in desired_order:
            desired_pos = desired_order.index(profile_id)
            # Weight differences more heavily for top positions
            position_weight = 1.0 / (i + 1)
            difference = abs(i - desired_pos)
            weighted_difference += difference * position_weight
    
    max_possible_weighted_difference = sum(1.0 / (i + 1) for i in range(total_positions)) * total_positions
    similarity = 1 - (weighted_difference / max_possible_weighted_difference)
    
    return similarity

def run(starting_profile: Profile):
    profile_objects = create_test_profiles()
    profile_list = assign_profiles_to_profile_list(starting_profile, profile_objects)
    
    # Define desired order
    desired_order = [167, 236, 340, 390, 384, 34, 96, 55]
    
    print("Initial weights:", {
        'age_similarity_weight': starting_profile.age_similarity_weight,
        'occupation_weight': starting_profile.occupation_weight,
        'work_industry_weight': starting_profile.work_industry_weight,
        'activity_hours_weight': starting_profile.activity_hours_weight,
        'smoking_weight': starting_profile.smoking_weight,
        'budget_weight': starting_profile.budget_weight,
        'origin_country_weight': starting_profile.origin_country_weight
    })
    
    # Optimize weights and get the best weights
    optimized_weights = optimize_weights(starting_profile, profile_list, desired_order)
    
    # Calculate final scores with optimized weights
    final_scores = []
    for profile in profile_list:
        score = calculate_overall_score(starting_profile, profile)
        final_scores.append((profile, score))
    
    # Sort and print final order
    final_order = [p.user_id for p, _ in sorted(final_scores, key=lambda x: x[1], reverse=True)]
    print("\nFinal order with optimized weights:", final_order)
    print("Desired order:", desired_order)
    
    # Continue with the rest of your code...

if __name__ == "__main__":
    run(starting_profile)