import os
import sys
import pandas as pd
import random

# Add the project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(project_root)

from app.rules_based.helper_functions import (
    PrintFunctions,
    Constants,
    initialize_profile_list_from_csv,
    modify_weights_with_weighted_average,
    assign_profiles_to_profile_list,
    calculate_overall_score,
    calculate_age,
)


def get_user_filters():
    """Get filter preferences from user input"""
    filters = {}

    print("\nWould you like to add any filters? (yes/no)")
    if input().lower() == "yes":
        print("\nAvailable filters:")
        print("1. Country")
        print("2. University")
        print("3. Course")
        print("4. Work Industry")
        print("5. Active Today")

        while True:
            print("\nEnter filter number (or 0 to finish):")
            choice = input()

            if choice == "0":
                break

            if choice == "1":
                country = input("Enter country code (e.g., GB, US): ").upper()
                filters["country"] = country
            elif choice == "2":
                university = input("Enter university (e.g., KCL, UCL): ").upper()
                filters["university"] = university
            elif choice == "3":
                course = input("Enter course: ")
                filters["course"] = course
            elif choice == "4":
                industry = input("Enter work industry: ")
                filters["work_industry"] = industry
            elif choice == "5":
                filters["active_today"] = True

    return filters


def run_test():
    # Initialize profiles from CSV
    csv_path = os.path.join(os.path.dirname(__file__), "big_boy_stuff.csv")
    all_profiles = initialize_profile_list_from_csv(csv_path)

    if not all_profiles:
        print("Failed to load profiles from CSV")
        return

    # Show available profiles before asking for input
    print("\nAvailable profiles to choose from:")
    for profile in all_profiles:
        print(
            f"ID: {profile.user_id}, Name: {profile.first_name} {profile.last_name}, university: {profile.university_id}, course: {profile.course}, work industry: {profile.work_industry}"
        )

    # Get starting profile by user ID
    profile_user_id = int(input("\nEnter the User ID for the starting profile: "))
    try:
        starting_profile = [p for p in all_profiles if p.user_id == profile_user_id][0]
    except IndexError:
        print(f"No profile found with ID: {profile_user_id}")
        return

    # Get user filters
    filters = get_user_filters()

    # Pass filters to assign_profiles_to_profile_list
    profile_list = assign_profiles_to_profile_list(
        starting_profile, all_profiles, filters=filters
    )

    print(f"Number of eligible profiles: {len(profile_list) if profile_list else 0}")

    # Show available profiles to choose from
    print("\nAvailable profiles to choose who liked the test user:")
    for profile in profile_list:
        print(
            f"ID: {profile.user_id}, Name: {profile.first_name} {profile.last_name}, university: {profile.university_id}, course: {profile.course}, work industry: {profile.work_industry}"
        )

    # Get number of profiles that liked the test user
    num_liked_me = int(
        input("\nEnter number of profiles that will like the test user: ")
    )

    # Let user pick which profiles liked them
    users_who_liked_me = []
    print("\nEnter the IDs of profiles that liked the test user:")
    while len(users_who_liked_me) < num_liked_me:
        try:
            user_id = int(
                input(
                    f"Enter ID for profile {len(users_who_liked_me) + 1}/{num_liked_me}: "
                )
            )
            if user_id in [p.user_id for p in profile_list]:
                if user_id not in users_who_liked_me:
                    users_who_liked_me.append(user_id)
                else:
                    print("This profile has already been selected.")
            else:
                print("Invalid ID. Please choose from the available profiles.")
        except ValueError:
            print("Please enter a valid number.")

    # Print original weights
    PrintFunctions.print_weights(starting_profile, "Initial")

    profiles_liked = 0
    while profiles_liked < 5:
        user_id = int(input("Enter the User ID for Alex to like: "))
        liked_profile = next((p for p in profile_list if p.user_id == user_id))
        if liked_profile:
            profiles_liked += 1
            starting_profile.likes.append(liked_profile)  # add days later

            if profiles_liked == 5:
                starting_profile = modify_weights_with_weighted_average(
                    starting_profile, Constants.LEARNING_RATE
                )
                PrintFunctions.print_weights(
                    starting_profile, f"After {len(starting_profile.likes)} Likes"
                )

    # Calculate scores for all profiles
    scored_profiles = []
    high_scoring_profiles = []
    liked_by_profiles = []

    for profile in profile_list:
        score = calculate_overall_score(starting_profile, profile)
        liked_by_me = any(
            liked.user_id == profile.user_id for liked in starting_profile.likes
        )
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
    high_scoring_profiles.sort(key=lambda x: x[1], reverse=True)

    # Create blocks with dynamic sizing
    final_profile_list = []
    base_block_size = Constants.BLOCK_SIZE  # 10

    for i in range(0, len(scored_profiles), base_block_size):
        # Calculate number of special profiles to add to this block (max 3 liked_by + 1 high_scoring)
        special_profiles_count = min(3, len(liked_by_profiles)) + min(
            1, len(high_scoring_profiles)
        )

        # Adjust block size based on number of special profiles to be added
        adjusted_block_size = base_block_size - special_profiles_count

        # Get initial block of adjusted size and shuffle
        block = scored_profiles[i : i + adjusted_block_size]
        random.shuffle(block)

        # Take first 5 positions of the block (or less if adjusted size is smaller)
        first_five = block[: min(5, len(block))]
        remaining_block = block[min(5, len(block)) :]

        # Add up to 3 liked_by profiles
        for _ in range(min(3, len(liked_by_profiles))):
            if liked_by_profiles:
                liked_profile = liked_by_profiles.pop(0)
                insert_position = random.randint(0, len(first_five) + 1)
                first_five.insert(insert_position, liked_profile)

        # Reconstruct the block
        block = first_five + remaining_block

        # Add high scoring profile if available
        if high_scoring_profiles:
            high_score_profile = high_scoring_profiles.pop(0)
            insert_position = random.randint(0, len(block) + 1)
            block.insert(insert_position, high_score_profile)

        final_profile_list.extend(block)

    # Add any remaining liked_by or high scoring profiles to the end of the list (TODO: not good solution)
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
        print(
            f"Position {i}: {profile.first_name} {profile.last_name} (ID: {profile.user_id}) - "
            f"Score: {score:.2f} - {position_type}{relationship_str}"
        )

    # Create sorted list for Excel
    all_profiles_sorted = sorted(final_profile_list, key=lambda x: x[1], reverse=True)

    # Create Excel file
    df = pd.DataFrame(
        [
            {
                "Rank": idx + 1,
                "User ID": profile.user_id,
                "Name": f"{profile.first_name} {profile.last_name}",
                "Compatibility Score": f"{score:.2f}",
                "Relationship": " & ".join(
                    [
                        "Liked by you" if liked_by_me else "",
                        "Liked you" if liked_me else "",
                    ]
                ).strip(),
                "Block Position": f"Block {i // 10 + 1}, {'first' if (i % 10) < 5 else 'last'} 5",
                "Age": calculate_age(profile.birth_date),
                "Gender": profile.gender,
                "Gender Living Pref": profile.sex_living_preference,
                "University": profile.university_id,
                "Course": profile.course,
                "Occupation": profile.occupation,
                "Work Industry": profile.work_industry,
                "Smoking": profile.smoking,
                "Activity Hours": profile.activity_hours,
                "Origin Country": profile.origin_country,
                "Budget Range": f"£{profile.rent_budget[0]}-£{profile.rent_budget[1]}",
            }
            for idx, (profile, score, liked_by_me, liked_me) in enumerate(
                all_profiles_sorted
            )
        ]
    )

    # Save to Excel
    df.to_excel("profiles_matched.xlsx", index=False)
    print("\nSorted profiles have been saved to 'profiles_matched.xlsx'")


if __name__ == "__main__":
    run_test()
