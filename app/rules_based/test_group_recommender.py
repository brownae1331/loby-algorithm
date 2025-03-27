import os
import sys
from helper_functions import initialize_profile_list_from_csv
from group_recommender import load_profile_matches, find_group_chat_recommendations
from typing import List

def main():
    # Set file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_path = os.path.join(base_dir, 'profiles-10.csv')  # Ensure this file exists
    matches_path = os.path.join(base_dir, 'profile_match-35.csv')
    
    # Check if files exist
    if not os.path.exists(profiles_path):
        print(f"Error: Profiles file not found at {profiles_path}")
        return
    
    if not os.path.exists(matches_path):
        print(f"Error: Matches file not found at {matches_path}")
        return
    
    # Load profiles 
    print("Loading profiles...")
    profiles = initialize_profile_list_from_csv(profiles_path)
    
    # Load profile matches - explicitly set to 7 days
    print("\nLoading profile matches...")
    profiles = load_profile_matches(matches_path, profiles, days=7)
    
    # Count profiles with matches
    profiles_with_matches = sum(1 for p in profiles if len(p.matches) > 0)
    print(f"Profiles with at least one match: {profiles_with_matches}")
    
    # Find potential group chat recommendations
    print("\nFinding potential group chat recommendations...")
    group_recommendations = find_group_chat_recommendations(profiles)
    
    # Display group recommendations
    if not group_recommendations:
        print("No group chat recommendations found.")
    else:
        print("\nPotential Group Chat Recommendations:")
        for i, group in enumerate(group_recommendations, 1):
            print(f"\nGroup {i}:")
            for profile in group['profiles']:
                print(f"  - {profile['name']} (ID: {profile['id']}, User ID: {profile['user_id']})")

if __name__ == "__main__":
    main() 

    ### Need to add fuction to check which group is most compatible and only send 1 notif at a time to a user 
    ### Need to add match time expiration for bridge recommendations (7 days - for now - will need to change if match number increases)