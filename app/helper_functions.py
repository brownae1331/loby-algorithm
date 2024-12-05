import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Optional, List, Union, Tuple, Any


from generate_profiles import Profile

# Define baseline threshold for significant preference
PREFERENCE_THRESHOLD = 0.5

class PrintFunctions:
    @staticmethod
    def print_weights(profile: Profile, title: str) -> None:
        print(f"\n{title} Weights:")
        print(f"Budget Weight: {profile.budget_weight}")
        print(f"Age Similarity Weight: {profile.age_similarity_weight}")
        print(f"Origin Country Weight: {profile.origin_country_weight}")
        print(f"Course Weight: {profile.course_weight}")
        print(f"Occupation Weight: {profile.occupation_weight}")
        print(f"Work Industry Weight: {profile.work_industry_weight}")
        print(f"Smoking Weight: {profile.smoking_weight}")
        print(f"Activity Hours Weight: {profile.activity_hours_weight}")

    @staticmethod
    def print_sorted_profiles_by_score(overall_scores_sorted: List[Tuple[Profile, float]]) -> None:
        print("\nProfiles sorted by compatibility scores:")
        for profile, score in overall_scores_sorted:
            print(f"User ID: {profile.user_id}, Name: {profile.first_name} {profile.last_name}, Score: {score}")


class CalculateScoreFunctions:
    @staticmethod
    def calculate_budget_overlap_score(starting_budget: Tuple[int, int], target_budget: Tuple[int, int]) -> float:
        """
        Calculate a budget overlap score between the starting profile budget and a target profile budget.
        The score is 1 if the budgets completely overlap, 0 if there's no overlap, and between 0 and 1 for partial overlap.
        """
        start_min, start_max = starting_budget
        target_min, target_max = target_budget
        
        # If there's no overlap
        if start_max < target_min or target_max < start_min:
            return 0.0
        
        # Calculate overlap length
        overlap_length = min(start_max, target_max) - max(start_min, target_min)
        budget_range = start_max - start_min
        
        return overlap_length / budget_range if budget_range > 0 else 0.0

    @staticmethod
    def calculate_age_similarity_score(starting_age: int, target_age: int) -> float:
        """
        Calculate an age similarity score between the starting profile age and a target profile age.
        The score is calculated using the function y = 1 - (1/15) * x^2, where y is the score and x is the age difference.
        """
        age_difference = abs(starting_age - target_age)
        return max(0.0, min(1.0, 1 - (age_difference ** 2) / 15))


class ComparisonFunctions:
    @staticmethod
    def compare_origin_country(attr1, attr2) -> float:
        """Compare origin country between profiles."""
        return float(attr1 == attr2)

    @staticmethod
    def compare_course(attr1, attr2) -> float:
        """Compare the course between profiles."""
        return float(attr1 == attr2)

    @staticmethod
    def compare_occupation(attr1, attr2) -> float:
        """Compare the occupation between profiles."""
        return float(attr1 == attr2)

    @staticmethod
    def compare_work_industry(attr1, attr2) -> float:
        """Compare the work industry between profiles."""
        return float(attr1 == attr2)

    @staticmethod
    def compare_smoking(attr1, attr2) -> float:
        """Compare smoking preferences between profiles."""
        return float(attr1 == attr2)

    @staticmethod
    def compare_activity_hours(attr1, attr2) -> float:
        """Compare activity hours between profiles."""
        return float(attr1 == attr2)


def calculate_age(birth_date: date) -> int:
    """Calculate age from birth date."""
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


def generate_likes(current_profile: Profile, profile_list: List[Profile]) -> None:
    print("\nAvailable profiles:")
    for profile in profile_list:
        budget_score = CalculateScoreFunctions.calculate_budget_overlap_score(current_profile.rent_budget, profile.rent_budget)
        age_score = CalculateScoreFunctions.calculate_age_similarity_score(calculate_age(current_profile.birth_date), calculate_age(profile.birth_date))
        country_score = ComparisonFunctions.compare_origin_country(current_profile.origin_country, profile.origin_country)
        smoking_score = ComparisonFunctions.compare_smoking(current_profile.smoking, profile.smoking)
        occupation_score = ComparisonFunctions.compare_occupation(current_profile.occupation, profile.occupation)
        industry_score = ComparisonFunctions.compare_work_industry(current_profile.work_industry, profile.work_industry)
        
        print(f"ID: {profile.user_id}, Budget: {budget_score:.2f}, Age: {age_score:.2f}, Country: {country_score:.1f}, Smoking: {smoking_score:.1f}, Occupation: {occupation_score:.1f}, Industry: {industry_score:.1f}")
    
    for i in range(5):
        user_id = int(input("Enter the User ID for Alex to like: "))
        # Find the profile with the given user ID
        liked_profile = next((p for p in profile_list if p.user_id == user_id), None)
        if liked_profile:
            current_profile.likes.append(liked_profile)
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

    return profile_objects


def modify_weights_with_weighted_average(current_profile: Profile, learning_rate: float = 0.1) -> Profile:
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
        'activity_hours_weight': 0.0
    }

    # Calculate average scores from liked profiles
    for liked_profile in current_profile.likes:
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
        avg_scores['smoking_weight'] += ComparisonFunctions.compare_smoking(
            current_profile.smoking, liked_profile.smoking)
        avg_scores['activity_hours_weight'] += ComparisonFunctions.compare_activity_hours(
            current_profile.activity_hours, liked_profile.activity_hours)

    # Calculate averages
    for key in avg_scores:
        avg_scores[key] /= total_likes

    # Adjust weights based on average scores
    for key in avg_scores:
        current_score = avg_scores[key]
        current_weight = getattr(current_profile, key)
        
        # Calculate weight adjustment
        if current_score > PREFERENCE_THRESHOLD:
            weight_adjustment = learning_rate * (current_score - PREFERENCE_THRESHOLD)
        else:
            weight_adjustment = -learning_rate * (PREFERENCE_THRESHOLD - current_score)

        # Apply weight adjustment with lower bounds
        new_weight = current_weight + weight_adjustment
        new_weight = max(0.05, min(2.0, new_weight))
        
        setattr(current_profile, key, new_weight)

    return current_profile


def calculate_overall_score(starting_profile: Profile, profile: Profile) -> float:
    """
    Calculate the overall score for a profile by calling all the comparison functions and summing their weighted scores.
    """
    budget_overlap_score: float = CalculateScoreFunctions.calculate_budget_overlap_score(starting_profile.rent_budget, profile.rent_budget) * profile.budget_weight
    age_similarity_score: float = CalculateScoreFunctions.calculate_age_similarity_score(calculate_age(starting_profile.birth_date), calculate_age(profile.birth_date)) * profile.age_similarity_weight
    origin_country_score: float = ComparisonFunctions.compare_origin_country(starting_profile.origin_country, profile.origin_country) * profile.origin_country_weight
    course_score: float = ComparisonFunctions.compare_course(starting_profile.course, profile.course) * profile.course_weight
    occupation_score: float = ComparisonFunctions.compare_occupation(starting_profile.occupation, profile.occupation) * profile.occupation_weight
    work_industry_score: float = ComparisonFunctions.compare_work_industry(starting_profile.work_industry, profile.work_industry) * profile.work_industry_weight
    smoking_score: float = ComparisonFunctions.compare_smoking(starting_profile.smoking, profile.smoking) * profile.smoking_weight
    activity_hours_score: float = ComparisonFunctions.compare_activity_hours(starting_profile.activity_hours, profile.activity_hours) * profile.activity_hours_weight
    
    overall_score: float = (
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
