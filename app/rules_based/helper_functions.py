import os
import sys
import random

# Check if the module is being run directly or being imported
if __name__ == '__main__':
    # Running directly - use absolute import
    from generate_profiles import Profile
else:
    # Being imported - use relative import
    from .generate_profiles import Profile

import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Union, Tuple, Any

from matplotlib.style.core import available
from countryinfo import CountryInfo

# Preference and learning constants
PREFERENCE_THRESHOLD = 0.5
DECAY_RATE = 0.7
LEARNING_RATE = 0.1

# Weight boundaries
MIN_WEIGHT = 0.005
MAX_WEIGHT = 2.0

# Age similarity constants
MIN_AGE_SCORE = -2.0
MAX_AGE_SCORE = 1.0
AGE_DIVISOR = 9

# Profile generation
NUM_PROFILES = 500

# Groups of compatible countries 
SPECIAL_COUNTRY_GROUPS = {
    'South_Asian': ['India', 'Pakistan', 'Bangladesh', 'Afghanistan', 'Nepal', 'Bhutan', 'Sri Lanka', 'Maldives'],
    'Arab': ['United Arab Emirates', 'Saudi Arabia', 'Qatar', 'Bahrain', 'Kuwait', 'Oman', 'Egypt', 'Algeria', 'Morocco', 'Tunisia', 'Libya', 'Jordan', 'Lebanon', 'Iraq', 'Syria', 'Sudan', 'Mauritania', 'Somalia', 'Djibouti', 'Comoros', 'Palestine', 'Yemen'],
    'East_Asian': ['China', 'Japan', 'South Korea', 'North Korea', 'Taiwan', 'Hong Kong', 'Mongolia', 'Macau'],
    'Anglosphere': ['United States', 'United Kingdom', 'Canada', 'Australia', 'New Zealand', 'Ireland'],
    'Eastern_Europe': ['Poland', 'Czech Republic', 'Slovakia', 'Hungary', 'Romania', 'Bulgaria', 'Ukraine', 'Belarus', 'Moldova', 'Russia', 'Latvia', 'Lithuania', 'Estonia', 'Serbia', 'Bosnia and Herzegovina', 'Montenegro', 'North Macedonia', 'Albania', 'Kosovo', 'Kazakhstan', 'Kyrgyzstan', 'Uzbekistan', 'Turkmenistan', 'Tajikistan']
}

def calculate_k_factor(time_difference_days, DECAY_RATE) -> float:
    """
    Calculate the k-factor based on the time difference from the liked_time.
    The k-factor decreases exponentially based on the decay_rate for each day since the liked_time.
    """
    k_factor = np.exp(-DECAY_RATE * time_difference_days)
    return k_factor

class PrintFunctions:
    @staticmethod
    def print_weights(profile: Profile, title: str) -> None:
        print(f"\n{title} Weights:")
        print(f"Budget Weight: {profile.budget_weight:.2f}")
        print(f"Age Similarity Weight: {profile.age_similarity_weight:.2f}")
        print(f"Origin Country Weight: {profile.origin_country_weight:.2f}")
        print(f"Course Weight: {profile.course_weight:.2f}")
        print(f"Occupation Weight: {profile.occupation_weight:.2f}")
        print(f"Work Industry Weight: {profile.work_industry_weight:.2f}")
        print(f"Smoking Weight: {profile.smoking_weight:.2f}")
        print(f"Activity Hours Weight: {profile.activity_hours_weight:.2f}")
        print(f"University Weight: {profile.university_weight:.2f}")

    @staticmethod
    def print_sorted_profiles_by_score(overall_scores_sorted: List[Tuple[Profile, float]]) -> None:
        print("\nProfiles sorted by compatibility scores:")
        for profile, score in overall_scores_sorted:
            print(f"User ID: {profile.user_id}, Name: {profile.first_name} {profile.last_name}, "
                  f"Living Pref: {profile.sex_living_preference}, Score: {score}")


class CalculateScoreFunctions:
    @staticmethod
    def calculate_budget_overlap_score(starting_budget: Tuple[int, int], target_budget: Tuple[int, int]) -> float:
        """
        Calculate a budget overlap score between the starting profile budget and a target profile budget.
        Returns:
        - 1.0: if budgets have significant overlap (>50%)
        - 0.5: if budgets have some overlap (>0% but ≤50%)
        - 0.0: if budgets have no overlap
        - -1.0: if either budget is None
        """
        if starting_budget is None or target_budget is None:
            return -1.0
            
        start_min, start_max = starting_budget
        target_min, target_max = target_budget
        
        # If there's no overlap
        if start_max < target_min or target_max < start_min:
            return 0.0
        
        # Calculate overlap length
        overlap_length = min(start_max, target_max) - max(start_min, target_min)
        budget_range = start_max - start_min
        
        if budget_range <= 0:
            return 0.0
            
        overlap_percentage = overlap_length / budget_range
        
        # Categorize into three levels
        if overlap_percentage > 0.5:
            return 1.0
        elif overlap_percentage > 0:
            return 0.5
        else:
            return 0.0

    @staticmethod
    def calculate_age_similarity_score(starting_age: int, target_age: int) -> float:
        """
        Calculate an age similarity score between the starting profile age and a target profile age.
        The score is calculated using the function y = 1 - (1/9) * x^2, where y is the score and x is the age difference.
        Final score is normalized by dividing by 3 to ensure it's between 0 and 1.
        """
        age_difference = abs(starting_age - target_age)
        score = max(-2.0, min(1.0, 1 - (age_difference ** 2) / 9))
        return (score + 2) / 3  # Normalize from [-2,1] to [0,1]


class ComparisonFunctions:
    @staticmethod
    def compare_origin_country(attr1: str, attr2: str) -> Tuple[float, bool]:
        """
        Compare origin countries and indicate if they're in a special group.
        Returns:
        - Tuple[float, bool]: (similarity_score, is_special_group)
        """
        if attr1 == attr2:
            # Check if the country is in any special group
            for group in SPECIAL_COUNTRY_GROUPS.values():
                if attr1 in group:
                    return 1.0, True
            return 1.0, False
            
        # Check if countries are in the same special group
        for group in SPECIAL_COUNTRY_GROUPS.values():
            if attr1 in group and attr2 in group:
                return 0.8, True
                
        return 0.0, False

    @staticmethod
    def compare_course(attr1, attr2) -> float:
        """Compare the course between profiles."""
        if attr1 is None or attr2 is None:
            return -1.0
        return float(attr1 == attr2)

    @staticmethod
    def compare_occupation(attr1, attr2) -> float:
        """Compare the occupation between profiles."""
        return float(attr1 == attr2)

    @staticmethod
    def compare_work_industry(attr1, attr2) -> float:
        """Compare the work industry between profiles."""
        if attr1 is None or attr2 is None:
            return -1.0
        return float(attr1 == attr2)

    @staticmethod
    def compare_smoking(attr1, attr2) -> float:
        """Compare smoking preferences between profiles."""
        if attr1 is None or attr2 is None:
            return -1.0
        return float(attr1 == attr2)

    @staticmethod
    def compare_activity_hours(attr1, attr2) -> float:
        """Compare activity hours between profiles."""
        return float(attr1 == attr2)

    @staticmethod
    def compare_university(attr1, attr2) -> float:
        """Compare the university between profiles."""
        if attr1 is None or attr2 is None:
            return -1.0
        return float(attr1 == attr2)

    @staticmethod
    def compare_gender(gender1: str, gender2: str) -> float:
        """Compare gender between profiles."""
        return float(gender1 == gender2)

def calculate_age(birth_date: date) -> int:
    """Calculate age from birth date."""
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


def generate_likes(starting_profile: Profile, current_profile: Profile, profile_list: List[Profile]) -> None:
    # Print initial weights
    PrintFunctions.print_weights(current_profile, "Initial")
    
    # Create a list to store profile data
    available_profiles_data = []
    
    print("\nAvailable profiles:")
    for profile in profile_list:
        # Print each profile's basic information
        print(f"ID: {profile.user_id}, "
              f"Age: {calculate_age(profile.birth_date)}, "
              f"Budget: £{profile.rent_budget[0]}-£{profile.rent_budget[1]}")
        
        # Add profile data to list for Excel export
        available_profiles_data.append({
            'ID': profile.user_id,
            'Name': f"{profile.first_name} {profile.last_name}",
            'Age': calculate_age(profile.birth_date),
            'Gender': profile.gender,
            'Budget': f"£{profile.rent_budget[0]}-£{profile.rent_budget[1]}",
            'Origin Country': profile.origin_country,
            'Occupation': profile.occupation,
            'Work Industry': profile.work_industry,
            'Course': profile.course,
            'Smoking': profile.smoking,
            'Activity Hours': profile.activity_hours,
            'Available From': profile.available_at
        })

    # Create DataFrame and export to Excel
    available_df = pd.DataFrame(available_profiles_data)
    available_df.to_excel('available_profiles.xlsx', index=False, engine='openpyxl')
    print("\nAvailable profiles have been exported to 'available_profiles.xlsx'")

    profiles_liked = 0
    while profiles_liked < 5:
        user_id = int(input("Enter the User ID for Alex to like: "))
        liked_profile = next((p for p in profile_list if p.user_id == user_id), None)
        if liked_profile:
            profiles_liked += 1
            days = int(input("Days: "))
            current_profile.likes.append([liked_profile, days])
            
            # When we reach 5 likes, update weights and recalculate scores
            if profiles_liked == 5:
                current_profile = modify_weights_with_weighted_average(current_profile, LEARNING_RATE)
                PrintFunctions.print_weights(current_profile, f"After {len(current_profile.likes)} Likes")
                
                # Calculate and export updated scores
                initial_scores = []
                overall_scores = []
                
                # Get initial scores
                for profile in profile_list:
                    initial_score = calculate_overall_score(starting_profile, profile)
                    initial_scores.append((profile, initial_score))
                
                # Get updated scores
                for profile in profile_list:
                    # Copy the updated weights
                    profile.budget_weight = current_profile.budget_weight
                    profile.age_similarity_weight = current_profile.age_similarity_weight
                    profile.origin_country_weight = current_profile.origin_country_weight
                    profile.course_weight = current_profile.course_weight
                    profile.occupation_weight = current_profile.occupation_weight
                    profile.work_industry_weight = current_profile.work_industry_weight
                    profile.smoking_weight = current_profile.smoking_weight
                    profile.activity_hours_weight = current_profile.activity_hours_weight
                    profile.university_weight = current_profile.university_weight
                    
                    overall_score = calculate_overall_score(current_profile, profile)
                    overall_scores.append((profile, overall_score))
                
                # Create dictionary mapping profile ID to initial score
                initial_scores_dict = {profile.user_id: score for profile, score in initial_scores}
                
                # Create profiles data while maintaining the order from profile_list
                profiles_data = []
                for profile in profile_list:  # Use profile_list to maintain order
                    score = next(score for p, score in overall_scores if p.user_id == profile.user_id)
                    profiles_data.append({
                        'Initial Score': round(initial_scores_dict[profile.user_id], 2),
                        'Updated Score': round(score, 2),
                        'Score Change': round(score - initial_scores_dict[profile.user_id], 2),
                        'Profile ID': profile.user_id,
                        'Name': f"{profile.first_name} {profile.last_name}",
                        'Age': calculate_age(profile.birth_date),
                        'Origin Country': profile.origin_country,
                        'Occupation': profile.occupation,
                        'Work Industry': profile.work_industry,
                        'Course': profile.course,
                        'Activity Hours': profile.activity_hours,
                        'Smoking': profile.smoking,
                        'Rent Budget': f"£{profile.rent_budget[0]}-£{profile.rent_budget[1]}",
                        'University': profile.university_id,
                    })
                
                # Create DataFrame without sorting
                results_df = pd.DataFrame(profiles_data)
                
                # Export to Excel
                results_df.to_excel('profile_matches.xlsx', index=False, engine='openpyxl')
                print("\nUpdated scores exported to profile_matches.xlsx")
                
                # Reset counter to allow for next batch
                profiles_liked = 0
        else:
            print(f"No profile found with User ID: {user_id}")


def assign_profiles_to_profile_list(starting_profile: Profile, profile_objects: List[Profile], profile_ids: Union[int, List[int]] = None) -> List[Profile]:
    """Assign profiles to the profile list based on move-in month, city, and sex_living preference"""
    profile_list: List[Profile] = []
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

    return profile_list


def initialize_profile_list() -> List[Profile]:
    np.random.seed(99)

    # Ensure all arrays have the same length
    num_profiles = 500

    # Generate a range of dates for the year 2024
    date_range = pd.date_range("2024-06-01", "2024-12-31")

    # First, generate occupations
    occupations = np.random.choice(["Student", "Working", "Cruising"], num_profiles)
    
    # Initialize universities array
    universities = np.array([None] * num_profiles)
    
    # Assign universities based on occupation
    student_mask = occupations == "Student"
    non_student_mask = ~student_mask
    
    # Students can only have actual universities (not 'none')
    universities[student_mask] = np.random.choice(["KCL", "UCL", "City", "QMU"], sum(student_mask))
    
    # Non-students can have any option including 'none'
    universities[non_student_mask] = np.random.choice(["KCL", "UCL", "City", "QMU", "none"], sum(non_student_mask))
    
    # Initialize arrays with None
    work_industries = np.array([None] * num_profiles)
    courses = np.array([None] * num_profiles)
    
    # Set values based on occupation
    working_mask = occupations == "Working"
    work_industries[working_mask] = np.random.choice(["Tech", "Finance", "Media"], sum(working_mask))
    
    student_mask = occupations == "Student"
    courses[student_mask] = np.random.choice(["Computer Science", "Business", "Engineering"], sum(student_mask))
    
    # Create DataFrame with updated logic
    profiles = pd.DataFrame({
        "user_id": range(1, num_profiles + 1),
        "first_name": np.random.choice(["John", "Jane", "Alice", "Bob"], num_profiles),
        "last_name": np.random.choice(["Doe", "Smith", "Johnson", "Williams"], num_profiles),
        "birth_date": pd.to_datetime(np.random.choice(pd.date_range("1997-01-01", "2006-12-31"), num_profiles)),
        "is_verified": np.random.choice([True, False], num_profiles),
        "gender": np.random.choice(["Male", "Female"], num_profiles),
        "description": np.random.choice([None, "Loves hiking", "Enjoys cooking", "Avid reader"], num_profiles),
        "languages": [np.random.choice(["English", "Spanish", "French", "Welsh"], np.random.randint(1, 4)).tolist() for _ in range(num_profiles)],
        "origin_country": np.random.choice(["UK", "USA", "Canada", "Australia"], num_profiles),
        "occupation": occupations,
        "work_industry": work_industries,
        "university_id": universities,
        "course": courses,
        "sexual_orientation": np.random.choice([None, "Heterosexual", "Homosexual", "Bisexual"], num_profiles),
        "pets": np.random.choice([None, "Dog", "Cat", "None"], num_profiles),
        "activity_hours": np.random.choice(["Morning", "Night"], num_profiles),
        "smoking": np.random.choice([None, "Yes", "No"], num_profiles),
        "extrovert_level": np.random.randint(1, 10, num_profiles),
        "cleanliness_level": np.random.randint(1, 10, num_profiles),
        "partying_level": np.random.randint(1, 10, num_profiles),
        "sex_living_preference": np.random.choice(["Male", "Female", "Both"], num_profiles),
        "rent_location_preference": np.random.choice(["London", "Bath", "Leeds", "Oxford"], num_profiles),
        "age_preference": [(18, 25) for _ in range(num_profiles)],
        "rent_budget": [(min_budget := np.random.randint(300, 1000), np.random.randint(min_budget, 2000)) for _ in range(num_profiles)],
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

    return profile_objects


def modify_weights_with_weighted_average(current_profile: Profile, LEARNING_RATE: float) -> Profile:
    """
    Modify weights using weighted averages and a learning rate.
    """
    total_likes = len(current_profile.likes)
    if total_likes == 0:
        return current_profile

    # Initialize average scores for each attribute
    avg_scores = {
        'budget_weight': 0.0,
        'age_similarity_weight': 0.0,
        'origin_country_weight': 0.0,
        'course_weight': 0.0,
        'occupation_weight': 0.0,
        'work_industry_weight': 0.0,
        'smoking_weight': 0.0,
        'activity_hours_weight': 0.0,
        'university_weight': 0.0,
        'gender_similarity_weight': 0.0,
    }

    # Calculate average scores from liked profiles
    for [liked_profile, days] in current_profile.likes:
        avg_scores['budget_weight'] += CalculateScoreFunctions.calculate_budget_overlap_score(
            current_profile.rent_budget, liked_profile.rent_budget)
        avg_scores['age_similarity_weight'] += CalculateScoreFunctions.calculate_age_similarity_score(
            calculate_age(current_profile.birth_date),
            calculate_age(liked_profile.birth_date))
        avg_scores['origin_country_weight'] += ComparisonFunctions.compare_origin_country(
            current_profile.origin_country, liked_profile.origin_country)
        avg_scores['course_weight'] += ComparisonFunctions.compare_course(
            current_profile.course, liked_profile.course)
        avg_scores['occupation_weight'] += ComparisonFunctions.compare_occupation(
            current_profile.occupation, liked_profile.occupation)
        avg_scores['work_industry_weight'] += ComparisonFunctions.compare_work_industry(
            current_profile.work_industry, liked_profile.work_industry)
        smoking_score = ComparisonFunctions.compare_smoking(current_profile.smoking, liked_profile.smoking)
        if smoking_score != -1:
            avg_scores['smoking_weight'] += smoking_score
        avg_scores['activity_hours_weight'] += ComparisonFunctions.compare_activity_hours(
            current_profile.activity_hours, liked_profile.activity_hours)
        avg_scores['university_weight'] += ComparisonFunctions.compare_university(
            current_profile.university_id, liked_profile.university_id)
        avg_scores['gender_similarity_weight'] += ComparisonFunctions.compare_gender(
            current_profile.gender, liked_profile.gender)

    # Calculate averages
    for key in avg_scores:
        avg_scores[key] /= total_likes

    # Adjust weights based on average scores
    for key in avg_scores:
        current_score = avg_scores[key]
        current_weight = getattr(current_profile, key)
        
        # Calculate weight adjustment and scale it by k_factor
        weight_adjustment = LEARNING_RATE * (current_score - PREFERENCE_THRESHOLD) * calculate_k_factor(days, DECAY_RATE)
        new_weight = current_weight + weight_adjustment
        new_weight = max(MIN_WEIGHT, min(MAX_WEIGHT, new_weight))
        setattr(current_profile, key, new_weight)

    return current_profile


def calculate_overall_score(starting_profile: Profile, profile: Profile) -> float:
    """
    Calculate the overall score for a profile by calling all the comparison functions and summing their weighted scores.
    """
    budget_overlap_score = CalculateScoreFunctions.calculate_budget_overlap_score(
        starting_profile.rent_budget, profile.rent_budget)
    if budget_overlap_score != -1:
        budget_overlap_score *= profile.budget_weight
    else:
        budget_overlap_score = 0.0
    
    age_similarity_score: float = CalculateScoreFunctions.calculate_age_similarity_score(calculate_age(starting_profile.birth_date), calculate_age(profile.birth_date)) * profile.age_similarity_weight
   # orgin country score is calculated by choosing which weight to apply 
    origin_country_score, is_special_group = ComparisonFunctions.compare_origin_country(
        starting_profile.origin_country, 
        profile.origin_country
    )
    origin_country_weight = profile.special_origin_country_weight if is_special_group else profile.origin_country_weight
    origin_country_score *= origin_country_weight
    
    course_score: float = ComparisonFunctions.compare_course(starting_profile.course, profile.course)
    if course_score != -1:
        course_score *= profile.course_weight
    else:
        course_score = 0.0
        
    occupation_score: float = ComparisonFunctions.compare_occupation(starting_profile.occupation, profile.occupation) * profile.occupation_weight
    
    work_industry_score: float = ComparisonFunctions.compare_work_industry(starting_profile.work_industry, profile.work_industry)
    if work_industry_score != -1:
        work_industry_score *= profile.work_industry_weight
    else:
        work_industry_score = 0.0
        
    smoking_score: float = ComparisonFunctions.compare_smoking(starting_profile.smoking, profile.smoking)
    if smoking_score != -1:
        smoking_score *= profile.smoking_weight
    else:
        smoking_score = 0.0
        
    activity_hours_score: float = ComparisonFunctions.compare_activity_hours(starting_profile.activity_hours, profile.activity_hours) * profile.activity_hours_weight
    university_score: float = ComparisonFunctions.compare_university(starting_profile.university_id, profile.university_id)
    if university_score != -1:
        university_score *= profile.university_weight
    else:
        university_score = 0.0
    
    gender_similarity_score: float = ComparisonFunctions.compare_gender(starting_profile.gender, profile.gender) * profile.gender_similarity_weight
    
    overall_score: float = (
        budget_overlap_score +
        age_similarity_score +
        origin_country_score +
        course_score +
        occupation_score +
        work_industry_score +
        smoking_score +
        activity_hours_score +
        university_score +
        gender_similarity_score
    )
    
    # Calculate the maximum possible score (sum of all weights)
    max_possible_score = (
        (profile.budget_weight if budget_overlap_score != -1 else 0) +
        profile.age_similarity_weight +
        profile.origin_country_weight +
        profile.course_weight +
        profile.occupation_weight +
        profile.work_industry_weight +
        profile.smoking_weight +
        profile.activity_hours_weight +
        profile.university_weight +
        profile.gender_similarity_weight
    )
    
    # Normalize the score between 0 and 1
    normalized_score = overall_score / max_possible_score if max_possible_score > 0 else 0.0
    
    return normalized_score


def get_age_based_weights(birth_date: date) -> dict:
    """
    Returns different weight configurations based on age groups.
    Age groups: 18-21 (students), 22-25 (early career), 26+ (professionals)
    """
    age = calculate_age(birth_date)
    
    # Get default weights from Profile class
    weights = {
        'budget_weight': Profile.budget_weight,
        'age_similarity_weight': Profile.age_similarity_weight,
        'origin_country_weight': Profile.origin_country_weight,
        'course_weight': Profile.course_weight,
        'occupation_weight': Profile.occupation_weight,
        'work_industry_weight': Profile.work_industry_weight,
        'smoking_weight': Profile.smoking_weight,
        'activity_hours_weight': Profile.activity_hours_weight,
        'university_weight': Profile.university_weight,
    }
    
    # Apply age-based multipliers to the base weights
    if 18 <= age <= 21:  # Student age group
        weights['budget_weight'] *= 1.5
        weights['university_weight'] *= 2.0
        weights['course_weight'] *= 1.5
        weights['work_industry_weight'] *= 0.5
        weights['occupation_weight'] *= 0.5
        
    elif 22 <= age <= 25:  # Early career
        weights['budget_weight'] *= 1.2
        weights['work_industry_weight'] *= 1.5
        weights['occupation_weight'] *= 1.5
        weights['university_weight'] *= 0.5
        weights['course_weight'] *= 0.8
        
    else:  # 26+ Professional
        weights['budget_weight'] *= 2.0
        weights['work_industry_weight'] *= 2.0
        weights['occupation_weight'] *= 2.0
        weights['university_weight'] *= 0.1
        weights['course_weight'] *= 0.5
        weights['activity_hours_weight'] *= 1.5
    
    return weights
