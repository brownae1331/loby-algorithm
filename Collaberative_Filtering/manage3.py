import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from helper_functions3 import load_profiles, load_apartments, load_enquiries, build_interaction_matrix, compute_user_similarity
from generate_profiles3 import Profile, Apartment, UserPropertyFilter
from typing import List, Dict, Tuple, Set
import os

class CollaborativeFilteringMatcher:
    def __init__(self, profiles, apartments, user_apt_filters):
        self.profiles = profiles
        self.apartments = apartments
        self.user_apt_filters = user_apt_filters
        
        # Create mappings for faster lookup
        self.user_id_to_index = {profile.user_id: i for i, profile in enumerate(self.profiles)}
        self.apartment_id_to_index = {apartment.id: i for i, apartment in enumerate(self.apartments)}
        self.index_to_user_id = {i: profile.user_id for i, profile in enumerate(self.profiles)}
        self.index_to_apartment_id = {i: apartment.id for i, apartment in enumerate(self.apartments)}
        
        # Build the user-item interaction matrix
        self.interaction_matrix = build_interaction_matrix(self.profiles, self.apartments)
        
        # Compute user similarity matrix
        self.user_similarity = compute_user_similarity(self.interaction_matrix)

def main():
    # Set file paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    profiles_path = os.path.join(base_dir, 'profiles_17-03.csv')
    properties_path = os.path.join(base_dir, 'properties_17-03.csv')
    enquiries_path = os.path.join(base_dir, 'enquires_17-03.csv')
    
    # Load all data
    profiles, user_apt_filters = load_profiles(profiles_path)
    apartments = load_apartments(properties_path)
    profiles = load_enquiries(enquiries_path, profiles)

    # Create matcher
    matcher = CollaborativeFilteringMatcher(profiles_w_enquiries, apartments, user_apt_filters)
    
    # Test with a few user IDs
    test_users = [profiles[0].user_id, profiles[1].user_id, profiles[2].user_id]
    
    for user_id in test_users:
        print(f"\nRecommendations for user {user_id}:")
        recommendations = matcher.recommend_roommates(user_id, top_n=3)
        
        if not recommendations:
            print("  No recommendations found.")
        else:
            for rec_id, score in recommendations:
                # Find the recommended user's profile
                rec_profile = None
                for profile in profiles:
                    if profile.user_id == rec_id:
                        rec_profile = profile
                        break
                
                if rec_profile:
                    print(f"  {rec_profile.first_name} {rec_profile.last_name} (ID: {rec_id}) - Similarity: {score:.4f}")
                else:
                    print(f"  User ID: {rec_id} - Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
