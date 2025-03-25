import pandas as pd
from helper_functions3 import load_profiles, load_apartments, load_enquiries, CollaborativeFilteringMatcher
from generate_profiles3 import Profile, Apartment, UserPropertyFilter
from typing import List, Dict, Tuple, Set
import os


def main():
    # Set file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_path = os.path.join(base_dir, 'profiles_17-03.csv')
    properties_path = os.path.join(base_dir, 'properties_17-03.csv')
    enquiries_path = os.path.join(base_dir, 'enquires_17-03.csv')
    
    # Load all data
    print("Loading profiles, apartments, and enquiries...")
    profiles, user_apt_filters = load_profiles(profiles_path)
    apartments = load_apartments(properties_path)
    profiles = load_enquiries(enquiries_path, profiles)
    
    # Create the matcher with filtered dataset
    print("\n=== CREATING COLLABORATIVE FILTERING MATCHER WITH FILTERED DATASET ===")
    matcher = CollaborativeFilteringMatcher(profiles, apartments, user_apt_filters, similarity_method="matrix_factorization")
    
    # Test with users who have enquiries
    user_with_enquiries = 303  # Known to have enquiries
    
    # Find a user without enquiries
    for profile in profiles:
        if len(profile.apt_likes) == 0:
            user_without_enquiries = profile.user_id
            break
    else:
        user_without_enquiries = 9999  # Fallback
    
    # Test users
    test_users = [user_with_enquiries, user_without_enquiries]
    
    # Print recommendations for both types of users
    for user_id in test_users:
        # Find user profile
        user_profile = next((p for p in profiles if p.user_id == user_id), None)
        if not user_profile:
            print(f"\nUser {user_id} not found in dataset")
            continue
        
        print(f"\n=== RECOMMENDATIONS FOR USER {user_id} ===")
        print(f"User: {user_profile.first_name} {user_profile.last_name}")
        if user_profile.university_id:
            print(f"University ID: {user_profile.university_id}")
        print(f"Gender: {user_profile.gender}")
        print(f"Preferred gender: {user_profile.preferred_gender}")
        print(f"Liked apartments: {len(user_profile.apt_likes)}")
        
        # Get recommendations
        recommendations = matcher.recommend_roommates(user_id, top_n=5)
        
        if not recommendations:
            print("No recommendations found.")
        else:
            for rank, (rec_id, score) in enumerate(recommendations, 1):
                # Find the recommended user's profile
                rec_profile = next((p for p in profiles if p.user_id == rec_id), None)
                
                if rec_profile:
                    print(f"\n  {rank}. {rec_profile.first_name} {rec_profile.last_name} (ID: {rec_id}) - Similarity: {score:.4f}")
                    if rec_profile.university_id:
                        print(f"     University ID: {rec_profile.university_id}")
                    print(f"     Gender: {rec_profile.gender}")
                    print(f"     Available from: {rec_profile.available_at}")
                    print(f"     Liked apartments: {len(rec_profile.apt_likes)}")
                    
                    # Show common likes
                    common_likes = set(user_profile.apt_likes).intersection(set(rec_profile.apt_likes))
                    if common_likes:
                        print(f"     Common liked apartments: {len(common_likes)}")
                else:
                    print(f"\n  {rank}. User ID: {rec_id} - Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
