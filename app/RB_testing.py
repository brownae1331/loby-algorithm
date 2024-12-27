import pandas as pd
from typing import List, Tuple
from helper_functions import PrintFunctions, generate_likes, modify_weights_with_weighted_average, calculate_overall_score, calculate_age, assign_profiles_to_profile_list
from generate_profiles import Profile
import openpyxl
import random

# Sam's profile (updated to match glop's exactly)
starting_profile = Profile(
    user_id=0,
    first_name="Sam",
    last_name="Example",
    birth_date=pd.to_datetime("2002-05-22"),
    is_verified=True,
    gender="Female",
    description="Enjoys outdoor activities and reading",
    languages=["English", "Spanish"],
    origin_country="UK",
    occupation="Student",
    work_industry=None,
    university_id="UCL",
    course="Business",
    sexual_orientation="Heterosexual",
    pets="Dog",
    activity_hours=None,
    smoking=None,
    extrovert_level=7,
    cleanliness_level=8,
    partying_level=4,
    sex_living_preference="Both",
    rent_location_preference="London",
    age_preference=(22, 30),
    rent_budget=(500, 600),
    last_filter_processed_at=pd.to_datetime("2023-06-01"),
    available_at="2024-09",
    roommate_count_preference=2,
    interests=["Reading", "Traveling", "Cooking"]
)

def create_test_profiles() -> List[Profile]:
    # Base template for all profiles
    base_profile = {
        "is_verified": True,
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
    
    # Updated profile variations to match the terminal output
    profile_variations = [
        {
            "user_id": 483,
            "first_name": "Jane",
            "last_name": "Doe",
            "birth_date": pd.to_datetime("2002-01-01"),  # Age 22
            "origin_country": "Canada",
            "occupation": "Student",
            "work_industry": None,
            "course": "Computer Science",
            "smoking": "No",
            "university_id": "KCL",
            "rent_budget": (463, 1939),
            "activity_hours": "Night",
            "gender": "Male"
        },
        {
            "user_id": 26,
            "first_name": "John",
            "last_name": "Johnson",
            "birth_date": pd.to_datetime("2004-01-01"),  # Age 20
            "origin_country": "Australia",
            "occupation": "Student",
            "work_industry": None,
            "course": "Computer Science",
            "smoking": None,
            "university_id": "QMU",
            "rent_budget": (568, 686),
            "activity_hours": "Night",
            "gender": "Male"
        },
        {
            "user_id": 142,
            "first_name": "Alice",
            "last_name": "Williams",
            "birth_date": pd.to_datetime("2000-01-01"),  # Age 24
            "origin_country": "USA",
            "occupation": "Student",
            "work_industry": None,
            "course": "Business",
            "smoking": "No",
            "university_id": "QMU",
            "rent_budget": (631, 1156),
            "activity_hours": "Morning",
            "gender": "Male"
        },
        {
            "user_id": 40,
            "first_name": "Alice",
            "last_name": "Johnson",
            "birth_date": pd.to_datetime("2002-01-01"),  # Age 22
            "origin_country": "USA",
            "occupation": "Working",
            "work_industry": "Tech",
            "course": None,
            "smoking": None,
            "university_id": "KCL",
            "rent_budget": (370, 563),
            "activity_hours": "Night",
            "gender": "Female"
        },
        {
            "user_id": 271,
            "first_name": "Bob",
            "last_name": "Johnson",
            "birth_date": pd.to_datetime("2001-01-01"),  # Age 23
            "origin_country": "USA",
            "occupation": "Cruising",
            "work_industry": None,
            "course": None,
            "smoking": None,
            "university_id": "City",
            "rent_budget": (450, 1416),
            "activity_hours": "Morning",
            "gender": "Male"
        },
        {
            "user_id": 498,
            "first_name": "Bob",
            "last_name": "Smith",
            "birth_date": pd.to_datetime("2006-01-01"),  # Age 18
            "origin_country": "Australia",
            "occupation": "Student",
            "work_industry": None,
            "course": "Business",
            "smoking": "Yes",
            "university_id": "QMU",
            "rent_budget": (412, 1194),
            "activity_hours": "Night",
            "gender": "Male"
        },
        {
            "user_id": 44,
            "first_name": "Jane",
            "last_name": "Williams",
            "birth_date": pd.to_datetime("2000-01-01"),  # Age 24
            "origin_country": "Canada",
            "occupation": "Student",
            "work_industry": None,
            "course": "Engineering",
            "smoking": "Yes",
            "university_id": "QMU",
            "rent_budget": (756, 1125),
            "activity_hours": "Night",
            "gender": "Female"
        },
        {
            "user_id": 309,
            "first_name": "Jane",
            "last_name": "Smith",
            "birth_date": pd.to_datetime("2004-01-01"),  # Age 20
            "origin_country": "USA",
            "occupation": "Cruising",
            "work_industry": None,
            "course": None,
            "smoking": "No",
            "university_id": "KCL",
            "rent_budget": (817, 1611),
            "activity_hours": "Night",
            "gender": "Male"
        },
        {
            "user_id": 446,
            "first_name": "John",
            "last_name": "Johnson",
            "birth_date": pd.to_datetime("2004-01-01"),  # Age 20
            "origin_country": "Canada",
            "occupation": "Working",
            "work_industry": "Finance",
            "course": None,
            "smoking": "No",
            "university_id": "KCL",
            "rent_budget": (433, 1882),
            "activity_hours": "Night",
            "gender": "Female"
        },
        {
            "user_id": 12,
            "first_name": "Alice",
            "last_name": "Doe",
            "birth_date": pd.to_datetime("2003-01-01"),  # Age 21
            "origin_country": "Canada",
            "occupation": "Working",
            "work_industry": "Finance",
            "course": None,
            "smoking": None,
            "university_id": "UCL",
            "rent_budget": (893, 1748),
            "activity_hours": "Night",
            "gender": "Female"
        }
    ]
    
    # Create profiles by combining base template with variations
    test_profiles = [
        Profile(
            **{**base_profile, 
               **{k: (None if v == "None" else v) for k, v in var.items()}}
        ) for var in profile_variations
    ]
    
    return test_profiles


def optimize_weights(starting_profile: Profile, profile_list: List[Profile], desired_order: List[int]) -> dict:
    """Optimize weights to match desired profile ordering"""
    default_weights = {
        'budget_weight': 0.156,
        'age_similarity_weight': 0.287,
        'origin_country_weight': 0.029,
        'course_weight': 0.169,
        'occupation_weight': 0.273,
        'work_industry_weight': 0.094,
        'smoking_weight': 0.03,
        'activity_hours_weight': 0.017
    }
    
    best_similarity = 0
    best_weights = default_weights.copy()
    
    print("Starting optimization...")
    print("Desired order:", desired_order)
    
    # Try different weight combinations
    for i in range(50000):  # Increased iterations
        # Generate weights with wider range and different strategy
        if i < 25000:
            # First half: try completely random weights
            new_weights = {k: random.uniform(0.001, 2.0) for k in default_weights.keys()}
        else:
            # Second half: make small adjustments to best weights found so far
            new_weights = {
                k: best_weights[k] * random.uniform(0.8, 1.2) 
                for k in default_weights
            }
        
        # Update the weight attributes
        for p in [starting_profile] + profile_list:
            for key, value in new_weights.items():
                setattr(p, key, value)
        
        # Calculate scores using the new weights
        scores = []
        for p in profile_list:
            score = calculate_overall_score(starting_profile, p)
            scores.append((p, score))
            
        current_order = [p.user_id for p, _ in sorted(scores, key=lambda x: x[1], reverse=True)]
        
        # Debug print every 1000 iterations
        if i % 1000 == 0:
            print(f"\nIteration {i}")
            print("Current weights:", {k: round(v, 3) for k, v in new_weights.items()})
            print("Current order:", current_order)
        
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
    desired_order = [483, 26, 142, 40, 271, 498, 44, 309, 446, 12]  
    
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