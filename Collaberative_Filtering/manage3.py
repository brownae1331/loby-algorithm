import pandas as pd
from helper_functions3 import load_profiles, load_apartments, load_enquiries, CollaborativeFilteringMatcher
from generate_profiles3 import Profile, Apartment, UserPropertyFilter
from typing import List, Dict, Tuple, Set
import os
import time


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
    
    # Create comparison between different similarity methods
    print("\n=== COMPARING DIFFERENT SIMILARITY METHODS ===")
    
    similarity_methods = ["jaccard", "cosine", "matrix_factorization"]
    recommendations_by_method = {}
    
    # Test with a few user IDs
    test_users = [287]
    
    for method in similarity_methods:
        print(f"\n--- EVALUATING {method.upper()} SIMILARITY METHOD ---")
        
        # Time the matcher creation
        start_time = time.time()
        matcher = CollaborativeFilteringMatcher(profiles, apartments, user_apt_filters, similarity_method=method)
        end_time = time.time()
        
        print(f"Time to build similarity matrix: {end_time - start_time:.2f} seconds")
        
        # Print matrix statistics
        print("\n--- MATRIX STATISTICS ---")
        print(f"Interaction matrix shape: {matcher.interaction_matrix.shape}")
        total_interactions = matcher.interaction_matrix.sum()
        density = total_interactions / (matcher.interaction_matrix.shape[0] * matcher.interaction_matrix.shape[1])
        print(f"Total interactions: {int(total_interactions)}")
        print(f"Matrix density: {density:.6f}")
        
        # Analyze similarity matrix
        non_zeros = (matcher.user_similarity > 0).sum()
        total_cells = matcher.user_similarity.size - matcher.user_similarity.shape[0]  # Exclude diagonal
        print(f"Non-zero similarities: {non_zeros} out of {total_cells} ({non_zeros/total_cells:.4%})")
        
        # Store recommendations for each user by method
        recommendations_by_method[method] = {}
        
        for user_id in test_users:
            recommendations = matcher.recommend_roommates(user_id, top_n=5)
            recommendations_by_method[method][user_id] = recommendations
    
    # Print comparison of recommendations
    print("\n=== RECOMMENDATION COMPARISON BY METHOD ===")
    
    for user_id in test_users:
        print(f"\nRecommendations for user {user_id}:")
        
        for method in similarity_methods:
            recommendations = recommendations_by_method[method][user_id]
            print(f"\n  {method.upper()} method:")
            
            if not recommendations:
                print("    No recommendations found.")
            else:
                for rank, (rec_id, score) in enumerate(recommendations, 1):
                    # Find the recommended user's profile
                    rec_profile = next((p for p in profiles if p.user_id == rec_id), None)
                    
                    if rec_profile:
                        print(f"    {rank}. {rec_profile.first_name} {rec_profile.last_name} (ID: {rec_id}) - Similarity: {score:.4f}")
                    else:
                        print(f"    {rank}. User ID: {rec_id} - Similarity: {score:.4f}")
    
    # Choose matrix_factorization as the preferred method for detailed output
    preferred_method = "matrix_factorization"
    print(f"\n=== DETAILED RESULTS USING {preferred_method.upper()} METHOD ===")
    
    matcher = CollaborativeFilteringMatcher(profiles, apartments, user_apt_filters, similarity_method=preferred_method)
    
    for user_id in test_users:
        print(f"\nDetailed recommendations for user {user_id}:")
        user_profile = next((p for p in profiles if p.user_id == user_id), None)
        
        if user_profile:
            print(f"  User: {user_profile.first_name} {user_profile.last_name}")
            if user_profile.university_id:
                print(f"  University ID: {user_profile.university_id}")
            print(f"  Gender: {user_profile.gender}")
            print(f"  Preferred gender: {user_profile.preferred_gender}")
            print(f"  Liked apartments: {len(user_profile.apt_likes)}")
        
        recommendations = matcher.recommend_roommates(user_id, top_n=13)
        
        if not recommendations:
            print("  No recommendations found.")
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
                    print(f"  {rank}. User ID: {rec_id} - Similarity: {score:.4f}")


if __name__ == "__main__":
    main()
