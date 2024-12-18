import pandas as pd
from typing import List, Tuple
from helper_functions import PrintFunctions, initialize_profile_list, generate_likes, modify_weights_with_weighted_average, assign_profiles_to_profile_list, calculate_overall_score, calculate_age
from generate_profiles import Profile
import openpyxl

# Alex's profile
starting_profile = Profile(
    user_id=0,
    first_name="Pooja",
    last_name="Example",
    birth_date=pd.to_datetime("2004-05-22"),
    is_verified=True,
    gender="Male",
    description="Enjoys outdoor activities and reading",
    languages=["English", "Spanish"],
    origin_country="UK",
    occupation="Student",
    work_industry=None,
    university_id="City",
    course= "Computer Science",  
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
    rent_budget=(400, 650),
    last_filter_processed_at=pd.to_datetime("2023-06-01"),
    available_at="2024-09",
    roommate_count_preference=2,
    interests=["Reading", "Traveling", "Cooking"]
)

def run():
    overall_scores: List[Tuple[Profile, float]] = []
    profile_objects = initialize_profile_list()
    profile_list = assign_profiles_to_profile_list(starting_profile, profile_objects)
    generate_likes(starting_profile, profile_list)
    # PrintFunctions.print_weights(starting_profile, "Initial")
    current_profile = modify_weights_with_weighted_average(starting_profile)
    # PrintFunctions.print_weights(current_profile, "Updated")

    for profile in profile_list:
        overall_score: float = calculate_overall_score(starting_profile, profile)
        overall_scores.append((profile, overall_score))

    # Sort profiles by overall score from highest to lowest
    overall_scores_sorted = sorted(overall_scores, key=lambda x: x[1], reverse=True)
    # PrintFunctions.print_sorted_profiles_by_score(overall_scores_sorted)

    # ALL CODE ABOVE HERE
    profiles_data = [{
        'Score': round(score, 2),
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
    } for profile, score in overall_scores_sorted]

    results_df = pd.DataFrame(profiles_data)

    # Export to Excel
    results_df.to_excel('profile_matches.xlsx', index=False, engine='openpyxl')
    # print("Results have been exported to 'profile_matches.xlsx'")

    # Print the database of profiles
    print(results_df)

if __name__ == "__main__":
    run()