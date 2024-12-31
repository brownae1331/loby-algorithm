import pandas as pd
from typing import List, Tuple
from helper_functions import PrintFunctions, initialize_profile_list, generate_likes, modify_weights_with_weighted_average, assign_profiles_to_profile_list, calculate_overall_score, calculate_age, CalculateScoreFunctions
from generate_profiles import Profile
import openpyxl

# Tom's profile
starting_profile = Profile(
    user_id=0,
    first_name="peepee_man",
    last_name="Example",
    birth_date=pd.to_datetime("2005-05-22"),
    is_verified=True,
    gender="Male",
    description="Enjoys outdoor activities and reading",
    languages=["English", "Spanish"],
    origin_country="UK",
    occupation="Student",
    work_industry=None,
    university_id="UCL",
    sexual_orientation="Heterosexual",
    pets="Dog",
    activity_hours= "Night",
    smoking="Yes",
    extrovert_level=7,
    cleanliness_level=8,
    partying_level=4,
    sex_living_preference="Both",
    rent_location_preference="London",
    age_preference=(22, 30),
    rent_budget=(900, 1100),
    last_filter_processed_at=pd.to_datetime("2023-06-01"),
    available_at="2024-09",
    roommate_count_preference=2,
    interests=["Reading", "Traveling", "Cooking"],
    course="Computer Science"
)

def run():
    overall_scores: List[Tuple[Profile, float]] = []
    profile_objects = initialize_profile_list()
    profile_list = assign_profiles_to_profile_list(starting_profile, profile_objects)
    
    # Calculate initial scores before likes
    initial_scores = []
    for profile in profile_list:
        initial_score = calculate_overall_score(starting_profile, profile)
        initial_scores.append((profile, initial_score))
    
    generate_likes(starting_profile, starting_profile, profile_list)



if __name__ == "__main__":
    run()