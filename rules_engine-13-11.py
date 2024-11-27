import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Union

from generate_profiles import Profile
from compatibility import CompatibilityCalculator

np.random.seed(42)

# Ensure all arrays have the same length
num_profiles = 500

# Generate a range of dates for the year 2024
date_range = pd.date_range("2024-06-01", "2024-12-31")


# Example profile data
profiles = pd.DataFrame({
    "user_id": range(1, num_profiles + 1),
    "first_name": np.random.choice(["John", "Jane", "Alice", "Bob"], num_profiles),
    "last_name": np.random.choice(["Doe", "Smith", "Johnson", "Williams"], num_profiles),
    "birth_date": pd.to_datetime(np.random.choice(pd.date_range("1995-01-01", "2006-12-31"), num_profiles)),
    "is_verified": np.random.choice([True, False], num_profiles),
    "gender": np.random.choice(["Male", "Female"], num_profiles),
    "description": np.random.choice([None, "Loves hiking", "Enjoys cooking", "Avid reader"], num_profiles),
    "languages": [np.random.choice(["English", "Spanish", "French", "Welsh"], np.random.randint(1, 4)).tolist() for _ in range(num_profiles)],
    "origin_country": np.random.choice(["UK", "USA", "Canada", "Australia"], num_profiles),
    "occupation": np.random.choice(["Student", "Working", "Cruising"], num_profiles),
    "work_industry": np.random.choice([None, "Tech", "Finance", "Media"], num_profiles),
    "university_id": np.random.choice([None, 1, 2, 3], num_profiles),
    "course": np.random.choice([None, "Computer Science", "Business", "Engineering"], num_profiles),
    "sexual_orientation": np.random.choice([None, "Heterosexual", "Homosexual", "Bisexual"], num_profiles),
    "pets": np.random.choice([None, "Dog", "Cat", "None"], num_profiles),
    "activity_hours": np.random.choice(["Morning", "Evening", "Night"], num_profiles),
    "smoking": np.random.choice([None, "Yes", "No"], num_profiles),
    "extrovert_level": np.random.randint(1, 10, num_profiles),
    "cleanliness_level": np.random.randint(1, 10, num_profiles),
    "partying_level": np.random.randint(1, 10, num_profiles),
    "sex_living_preference": np.random.choice(["Male", "Female", "Both"], num_profiles),
    "rent_location_preference": np.random.choice(["London", "Bath", "Leeds", "Oxford"], num_profiles),
    "age_preference": [(18, 25) for _ in range(num_profiles)],
    "rent_budget": [(np.random.randint(500, 700), np.random.randint(700, 1000)) for _ in range(num_profiles)],
    "last_filter_processed_at": pd.to_datetime(np.random.choice(pd.date_range("2023-01-01", "2023-12-31"), num_profiles)),
    "available_at": np.random.choice(date_range.strftime('%Y-%m'), num_profiles),
    "roommate_count_preference": np.random.choice([1, 2, 3], num_profiles),
    "interests": [np.random.choice(["Reading", "Traveling", "Cooking", "Sports"], np.random.randint(1, 4)).tolist() for _ in range(num_profiles)]
})

# Convert DataFrame rows to Profile objects
profile_objects = [
    Profile(
        user_id=row["user_id"],
        first_name=row["first_name"],
        last_name=row["last_name"],
        birth_date=row["birth_date"],
        is_verified=row["is_verified"],
        gender=row["gender"],
        description=row["description"],
        languages=row["languages"],
        origin_country=row["origin_country"],
        occupation=row["occupation"],
        work_industry=row["work_industry"],
        university_id=row["university_id"],
        course=row["course"],
        sexual_orientation=row["sexual_orientation"],
        pets=row["pets"],
        activity_hours=row["activity_hours"],
        smoking=row["smoking"],
        extrovert_level=row["extrovert_level"],
        cleanliness_level=row["cleanliness_level"],
        partying_level=row["partying_level"],
        sex_living_preference=row["sex_living_preference"],
        rent_location_preference=row["rent_location_preference"],
        age_preference=row["age_preference"],
        rent_budget=row["rent_budget"],
        last_filter_processed_at=row["last_filter_processed_at"],
        available_at=row["available_at"],
        roommate_count_preference=row["roommate_count_preference"],
        interests=row["interests"]
    )
    for _, row in profiles.iterrows()
]

# Define a starting profile for comparison
starting_profile = Profile(
    user_id=0,
    first_name="Alex",
    last_name="Example",
    birth_date=pd.to_datetime("1998-05-22"),
    is_verified=True,
    gender="Male",
    description="Enjoys outdoor activities and reading",
    languages=["English", "Spanish"],
    origin_country="UK",
    occupation="Working",
    work_industry="Tech",
    university_id=1,
    course="Computer Science",
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


def calculate_age(birth_date):
    """Calculate age from birth date."""
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


# Define clusters as a global variable
profile_list = []

def assign_profiles_to_profile_list(profile_ids: Union[int, List[int]] = None):
    """Assign profiles to the profile list based on move-in month, city, and sex_living preference"""
    global profile_list
    if profile_ids is None:
        profile_ids = [profile.user_id for profile in profile_objects]
    elif isinstance(profile_ids, int):
        profile_ids = [profile_ids]

    # Convert starting profile's available_at to datetime for comparison
    starting_available_at = pd.to_datetime(starting_profile.available_at, format='%Y-%m')
    for profile in profile_objects:
        if profile.user_id in profile_ids:
            profile_available_at = pd.to_datetime(profile.available_at, format='%Y-%m')
            # Check if the profile's available_at is within one month earlier or later of starting profile's available_at
            if (starting_available_at - timedelta(days=31)) <= profile_available_at <= (starting_available_at + timedelta(days=31)) and profile.rent_location_preference == starting_profile.rent_location_preference:
                # Check if the gender matches the sex living preferences
                if (starting_profile.sex_living_preference == "Both" or starting_profile.sex_living_preference == profile.gender) and \
                   (profile.sex_living_preference == "Both" or profile.sex_living_preference == starting_profile.gender):
                    profile_list.append(profile)
        
assign_profiles_to_profile_list()
print(f"Number of profiles assigned: {len(profile_list)}")

# Display details of assigned profiles to verify
for profile in profile_list:  # Display only the first 5 profiles for brevity
    age = calculate_age(profile.birth_date)
    print(f"User ID: {profile.user_id}, Age: {age}, Budget Range: {profile.rent_budget}, Move-in Date: {profile.available_at}, City: {profile.rent_location_preference}, Gender: {profile.gender}, Gender_living_preference: {profile.sex_living_preference}")
    
    
def calculate_budget_overlap_score(starting_budget, target_budget):
    """
    Calculate a budget overlap score between the starting profile budget and a target profile budget.
    The score is 1 if the budgets completely overlap, 0 if there's no overlap, and between 0 and 1 for partial overlap.
    """
    start_min, start_max = starting_budget
    target_min, target_max = target_budget
    
    # Calculate overlap range
    overlap_min = max(start_min, target_min)
    overlap_max = min(start_max, target_max)
    
    # If there's no overlap
    if overlap_min > overlap_max:
        return 0.0
    
    # Calculate overlap length and total possible range
    overlap_length = overlap_max - overlap_min
    total_length = max(start_max, target_max) - min(start_min, target_min)
    
    # Return score based on overlap proportion
    return overlap_length / (start_max - start_min) if start_max - start_min > 0 else 0.0

# Calculate and display budget overlap scores for all profiles in profile_list
print("\nBudget Overlap Scores:\n")
for profile in profile_list:
    score = calculate_budget_overlap_score(starting_profile.rent_budget, profile.rent_budget)
    print(f"User ID: {profile.user_id}, Budget Range: {profile.rent_budget}, Budget Overlap Score: {score:.2f}")
    
    
def calculate_age_similarity_score(starting_age, target_age):
    """
    Calculate an age similarity score between the starting profile age and a target profile age.
    The score is calculated using the function y = 1 - (1/15) * x^2, where y is the score and x is the age difference.
    """
    age_difference = abs(starting_age - target_age)
    score = 1 - (1 / 15) * (age_difference ** 2)
    return max(0.0, min(1.0, score))  # Ensure the score is between 0 and 1

# Calculate and display age similarity scores for all profiles in profile_list
print("\nAge Similarity Scores:\n")
starting_profile_age = calculate_age(starting_profile.birth_date)
for profile in profile_list:
    target_profile_age = calculate_age(profile.birth_date)
    age_similarity_score = calculate_age_similarity_score(starting_profile_age, target_profile_age)
    print(f"User ID: {profile.user_id}, Age: {target_profile_age}, Age Similarity Score: {age_similarity_score:.2f}")

def compare_origin_country(starting_origin_country, target_origin_country):
    """Compare origin country between profiles."""
    if starting_origin_country == target_origin_country:
        return 1
    return 0
    
def compare_course(starting_course, target_course):
    """Compare the course between profiles."""
    if starting_course == target_course:
        return 1
    return 0

def compare_occupation(starting_occupation, target_occupation):
    """Compare the occupation between profiles."""
    if starting_occupation == target_occupation:
        return 1
    return 0
    
def compare_work_industry(starting_work_industry, target_work_industry):
    """Compare the work industry between profiles."""
    if starting_work_industry == target_work_industry:
        return 1
    return 0

def compare_smoking(starting_smoking, target_smoking):
    if starting_smoking == target_smoking:
        return 1
    return 0

def compare_activity_hours(starting_activity_hours, target_activity_hours):
    if starting_activity_hours == target_activity_hours:
        return 1
    return 0


def calculate_overall_score(profile):
    """
    Calculate the overall score for a profile by calling all the comparison functions and summing their weighted scores.
    """
    budget_overlap_score = calculate_budget_overlap_score(starting_profile.rent_budget, profile.rent_budget) * 4
    age_similarity_score = calculate_age_similarity_score(calculate_age(starting_profile.birth_date), calculate_age(profile.birth_date)) * 4
    origin_country_score = compare_origin_country(starting_profile.origin_country, profile.origin_country) * 1.4
    course_score = compare_course(starting_profile.course, profile.course) * 1.3
    occupation_score = compare_occupation(starting_profile.occupation, profile.occupation) * 1.5
    work_industry_score = compare_work_industry(starting_profile.work_industry, profile.work_industry) * 1.3
    smoking_score = compare_smoking(starting_profile.smoking, profile.smoking) * 1.2
    activity_hours_score = compare_activity_hours(starting_profile.activity_hours, profile.activity_hours) * 1
    
    overall_score = (
        budget_overlap_score +
        age_similarity_score +
        origin_country_score +
        course_score +
        occupation_score +
        work_industry_score +
        smoking_score +
        activity_hours_score
    )
    
    return overall_score

# Calculate and display overall scores for all profiles in profile_list
print("\nOverall Scores:\n")
overall_scores = []
for profile in profile_list:
    overall_score = calculate_overall_score(profile)
    overall_scores.append((profile, overall_score))
    print(f"User ID: {profile.user_id}, Overall Score: {overall_score:.2f}")
    
# Sort profiles by overall score from highest to lowest
overall_scores_sorted = sorted(overall_scores, key=lambda x: x[1], reverse=True)

# Display sorted profiles
print("\nProfiles sorted by Overall Score:\n")
for profile, score in overall_scores_sorted:
    print(f"User ID: {profile.user_id}, Overall Score: {score:.2f}")
    