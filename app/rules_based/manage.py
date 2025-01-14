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
    
    # Add users_who_liked_me list - in real implementation, this would come from your database
    users_who_liked_me = []  # TODO: Get this from your database
    likes_counter = 0
    weights_adjusted = False

    while profile_list:
        scored_profiles = []
        high_scoring_profiles = []
        liked_by_profiles = []  # New list for users who have liked us
        
        # Score and categorize profiles
        for profile in profile_list:
            score = calculate_overall_score(starting_profile, profile)
            
            # Check if this profile has liked us
            if profile.user_id in users_who_liked_me:
                score *= 1.5  # Boost score by 50%
                liked_by_profiles.append((profile, score))
            elif score >= 9:
                high_scoring_profiles.append((profile, score))
            else:
                scored_profiles.append((profile, score))
        
        # Sort all profile lists by score
        liked_by_profiles.sort(key=lambda x: x[1], reverse=True)
        scored_profiles.sort(key=lambda x: x[1], reverse=True)
        
        # Create blocks of 10 and randomize each block
        final_profile_list = []
        block_size = 10
        
        for i in range(0, len(scored_profiles), block_size):
            block = scored_profiles[i:i + block_size]
            random.shuffle(block)
            
            # Take first 5 positions of the block
            first_five = block[:5]
            remaining_block = block[5:]
            
            # Add 3 liked_by profiles in the first 5 positions if available
            liked_positions = random.sample(range(5), min(3, len(liked_by_profiles)))
            for pos in sorted(liked_positions, reverse=True):
                if liked_by_profiles:
                    liked_profile = liked_by_profiles.pop(0)
                    first_five.insert(pos, liked_profile)
            
            # Reconstruct the block
            block = first_five + remaining_block
            
            # Add high scoring profile if available
            if high_scoring_profiles:
                high_score_profile = high_scoring_profiles.pop(0)
                insert_position = random.randint(0, len(block))
                block.insert(insert_position, high_score_profile)
            
            final_profile_list.extend(block)
        
        # Add any remaining liked_by or high scoring profiles
        if liked_by_profiles:
            final_profile_list.extend(liked_by_profiles)
        if high_scoring_profiles:
            final_profile_list.extend(high_scoring_profiles)
        
        # Update profile_list with the new ordering
        profile_list = [profile for profile, _ in final_profile_list]
        
        # Process profiles and handle likes
        liked_profiles = generate_likes(starting_profile, starting_profile, profile_list)
        
        # Remove liked profiles from profile_list
        profile_list = [p for p in profile_list if p not in liked_profiles]
        
        # Update likes counter and weights
        likes_counter += len(liked_profiles)
        
        # Only break to rerank if we haven't adjusted weights yet and have 5 or more likes
        if not weights_adjusted and likes_counter >= 5:
            weights_adjusted = True  # Set flag to true after first adjustment
            likes_counter = 0  # Reset counter



if __name__ == "__main__":
    run()