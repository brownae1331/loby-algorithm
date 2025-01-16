import os
import sys
import pandas as pd
from typing import List, Tuple

# Add the project root to Python path when running directly
if __name__ == '__main__':
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(project_root)

from app.rules_based.helper_functions import (
    PrintFunctions, 
    initialize_profile_list, 
    generate_likes, 
    modify_weights_with_weighted_average, 
    assign_profiles_to_profile_list, 
    calculate_overall_score, 
    calculate_age, 
    CalculateScoreFunctions
)
from generate_profiles import Profile
import openpyxl
import random

def run_test():
    # Move starting_profile definition inside the function
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
        occupation="Student",
        work_industry=None,
        university_id="UCL",
        sexual_orientation="Heterosexual",
        pets="Dog",
        activity_hours="Night",
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

    # Initialize profiles
    profile_objects = initialize_profile_list()
    
    # Get eligible profiles first
    profile_list = assign_profiles_to_profile_list(starting_profile, profile_objects)
    
    # Show available profiles to choose from
    print("\nAvailable profiles to choose who liked the test user:")
    for profile in profile_list:
        print(f"ID: {profile.user_id}, Name: {profile.first_name} {profile.last_name}")
    
    # Get number of profiles that liked the test user
    num_liked_me = int(input("\nEnter number of profiles that will like the test user: "))
    
    # Let user pick which profiles liked them
    users_who_liked_me = []
    print("\nEnter the IDs of profiles that liked the test user:")
    while len(users_who_liked_me) < num_liked_me:
        try:
            user_id = int(input(f"Enter ID for profile {len(users_who_liked_me) + 1}/{num_liked_me}: "))
            if user_id in [p.user_id for p in profile_list]:
                if user_id not in users_who_liked_me:
                    users_who_liked_me.append(user_id)
                else:
                    print("This profile has already been selected.")
            else:
                print("Invalid ID. Please choose from the available profiles.")
        except ValueError:
            print("Please enter a valid number.")
    
    # Store original weights
    original_weights = {
        'budget_weight': starting_profile.budget_weight,
        'age_similarity_weight': starting_profile.age_similarity_weight,
        'origin_country_weight': starting_profile.origin_country_weight,
        'course_weight': starting_profile.course_weight,
        'occupation_weight': starting_profile.occupation_weight,
        'work_industry_weight': starting_profile.work_industry_weight,
        'smoking_weight': starting_profile.smoking_weight,
        'activity_hours_weight': starting_profile.activity_hours_weight,
        'university_weight': starting_profile.university_weight,
        'gender_similarity_weight': starting_profile.gender_similarity_weight,
    }
    
    # Print original weights
    print("\nOriginal Weights:")
    PrintFunctions.print_weights(starting_profile, "Initial")
    
    # Modify generate_likes to not print weights
    for _ in range(5):
        user_id = int(input("Enter the User ID for Alex to like: "))
        liked_profile = next((p for p in profile_list if p.user_id == user_id), None)
        if liked_profile:
            days = int(input("Days: "))
            starting_profile.likes.append([liked_profile, days])
    
    # Update weights after all likes
    starting_profile = modify_weights_with_weighted_average(starting_profile, 0.1)
    
    # Print updated weights and changes
    print("\nWeight Changes:")
    for attr, original in original_weights.items():
        new_value = getattr(starting_profile, attr)
        change = new_value - original
        print(f"{attr}: {original:.2f} -> {new_value:.2f} (Change: {change:+.2f})")
    
    # Calculate scores for all profiles
    scored_profiles = []
    high_scoring_profiles = []
    liked_by_profiles = []
    
    for profile in profile_list:
        score = calculate_overall_score(starting_profile, profile)
        liked_by_me = any(liked.user_id == profile.user_id for liked, _ in starting_profile.likes)
        liked_me = profile.user_id in users_who_liked_me
        
        profile_data = (profile, score, liked_by_me, liked_me)
        
        if liked_me:
            liked_by_profiles.append(profile_data)
        elif score >= 0.8:  # High scoring threshold
            high_scoring_profiles.append(profile_data)
        else:
            scored_profiles.append(profile_data)
    
    # Sort all lists by score
    liked_by_profiles.sort(key=lambda x: x[1], reverse=True)
    scored_profiles.sort(key=lambda x: x[1], reverse=True)
    
    # Create blocks of 10 with randomization
    final_profile_list = []
    block_size = 10
    
    for i in range(0, len(scored_profiles), block_size):
        block = scored_profiles[i:i + block_size]
        random.shuffle(block)
        
        # Take first 5 positions of the block
        first_five = block[:5]
        remaining_block = block[5:]
        
        # Add up to 3 liked_by profiles in the first 5 positions
        liked_positions = random.sample(range(5), min(3, len(liked_by_profiles)))
        for pos in sorted(liked_positions, reverse=True):
            if liked_by_profiles:
                liked_profile = liked_by_profiles.pop(0)
                first_five.insert(pos, liked_profile)
        
        # Ensure first_five only has 5 elements
        first_five = first_five[:5]
        
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
    
    # Print final list with blocks clearly marked
    print("\nFinal Profile List (in blocks of 10):")
    print("Format: Position. Name (ID) - Score [Relationship]")
    print("-" * 70)
    
    for i, (profile, score, liked_by_me, liked_me) in enumerate(final_profile_list):
        if i % 10 == 0:
            print(f"\nBlock {i // 10 + 1}:")
            print("-" * 20)
        
        relationship = []
        if liked_by_me:
            relationship.append("Liked by you")
        if liked_me:
            relationship.append("Liked you")
        relationship_str = f" [{' & '.join(relationship)}]" if relationship else ""
        
        position_type = "first 5" if (i % 10) < 5 else "last 5"
        print(f"Position {i}: {profile.first_name} {profile.last_name} (ID: {profile.user_id}) - "
              f"Score: {score:.2f} - {position_type}{relationship_str}")
    
    # Create sorted list for Excel
    all_profiles_sorted = sorted(final_profile_list, key=lambda x: x[1], reverse=True)
    
    # Create Excel file
    df = pd.DataFrame([{
        'Rank': idx + 1,
        'User ID': profile.user_id,
        'Name': f"{profile.first_name} {profile.last_name}",
        'Compatibility Score': f"{score:.2f}",
        'Relationship': ' & '.join(['Liked by you' if liked_by_me else '', 'Liked you' if liked_me else '']).strip(),
        'Block Position': f"Block {i//10 + 1}, {'first' if (i%10) < 5 else 'last'} 5",
        'Age': calculate_age(profile.birth_date),
        'Gender': profile.gender,
        'Gender Living Pref': profile.sex_living_preference,
        'University': profile.university_id,
        'Course': profile.course,
        'Occupation': profile.occupation,
        'Work Industry': profile.work_industry,
        'Smoking': profile.smoking,
        'Activity Hours': profile.activity_hours,
        'Origin Country': profile.origin_country,
        'Budget Range': f"£{profile.rent_budget[0]}-£{profile.rent_budget[1]}"
    } for idx, (profile, score, liked_by_me, liked_me) in enumerate(all_profiles_sorted)])
    
    # Save to Excel
    df.to_excel('profiles_matched.xlsx', index=False)
    print("\nSorted profiles have been saved to 'profiles_matched.xlsx'")

if __name__ == "__main__":
    run_test()