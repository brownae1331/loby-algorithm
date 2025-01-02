import pandas as pd
from typing import List, Tuple
from helper_functions import PrintFunctions, initialize_profile_list, generate_likes, modify_weights_with_weighted_average, assign_profiles_to_profile_list, calculate_overall_score, calculate_age, CalculateScoreFunctions
from generate_profiles import Profile
import openpyxl
import random

# Tom's profile
starting_profile = Profile(
    user_id=0,
    first_name="Christian",
    last_name="Example",
    birth_date=pd.to_datetime("2002-02-26"),
    is_verified=True,
    gender="Male",
    description="Enjoys outdoor activities and reading",
    languages=["English"],
    origin_country="UK",
    occupation="Cruising",
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
    age_preference=(20, 30),
    rent_budget=(500, 1100),
    last_filter_processed_at=pd.to_datetime("2023-06-01"),
    available_at="2024-09",
    roommate_count_preference=2,
    interests=["Reading", "Traveling", "Cooking"],
    course="Computer Science"
)

def run():
    profile_objects = initialize_profile_list()
    profile_list = assign_profiles_to_profile_list(starting_profile, profile_objects)
    
    # Calculate initial scores and create tuples of (profile, score)
    scored_profiles = []
    high_scoring_profiles = []
    for profile in profile_list:
        score = calculate_overall_score(starting_profile, profile)
        if score >= 9:
            high_scoring_profiles.append((profile, score))
        else:
            scored_profiles.append((profile, score))
    
    # Sort profiles by score
    scored_profiles.sort(key=lambda x: x[1], reverse=True)
    
    # Create blocks of 10 and randomize each block
    final_profile_list = []
    block_size = 10
    
    for i in range(0, len(scored_profiles), block_size):
        # Get the current block
        block = scored_profiles[i:i + block_size]
        # Randomize the block
        random.shuffle(block)
        
        # Insert a high-scoring profile if available
        if high_scoring_profiles:
            high_score_profile = high_scoring_profiles.pop(0)
            insert_position = random.randint(0, len(block))
            block.insert(insert_position, high_score_profile)
            
        final_profile_list.extend(block)
    
    # If any high-scoring profiles remain, add them to the end
    final_profile_list.extend(high_scoring_profiles)
    
    # Update profile_list with the new ordering
    profile_list = [profile for profile, _ in final_profile_list]
    
    generate_likes(starting_profile, starting_profile, profile_list)



if __name__ == "__main__":
    run()