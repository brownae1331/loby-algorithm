import pandas as pd
from datetime import datetime
from manage2 import XGBoostRecommender
from generate_profiles2 import Profile
import os

def load_data():
    """Load profile and likes data from CSV files"""
    # Use absolute path to find the CSV files
    current_dir = os.path.dirname(__file__)
    
    # Load and print column names to debug
    profiles_df = pd.read_csv(os.path.join(current_dir, 'profile.csv'))
    likes_df = pd.read_csv(os.path.join(current_dir, 'profile_like.csv'))
    
    print("Profile columns:", profiles_df.columns.tolist())
    print("Likes columns:", likes_df.columns.tolist())
    
    # Rename columns to match what the code expects
    likes_df = likes_df.rename(columns={
        'profile_id_1': 'user_id',
        'profile_id_2': 'liked_user_id'
    })
    
    return profiles_df, likes_df

def create_profile(row):
    """Create Profile object from profile dataframe row"""
    try:
        birth_date = datetime.strptime(str(row['birth_date']), '%Y-%m-%d').date()
    except:
        birth_date = datetime.now().date()  # fallback if date is invalid
        
    # Convert rent budget to tuple with default values if missing
    rent_budget = (800, 1200)  # default budget
    if pd.notna(row.get('rent_budget_min')) and pd.notna(row.get('rent_budget_max')):
        rent_budget = (int(row['rent_budget_min']), int(row['rent_budget_max']))
    
    # Convert age preference to tuple
    age_pref_min = row.get('age_preference_min', 18)
    age_pref_max = row.get('age_preference_max', 100)
    age_preference = (int(age_pref_min), int(age_pref_max))

    return Profile(
        user_id=int(row['user_id']),
        first_name=str(row.get('first_name', '')),
        last_name=str(row.get('last_name', '')),
        birth_date=birth_date,
        is_verified=bool(row.get('is_verified', False)),
        gender=str(row.get('gender', '')),
        description=str(row.get('description', None)),
        languages=row.get('languages', None),
        origin_country=str(row.get('origin_country', '')),
        occupation=str(row.get('occupation', '')),
        work_industry=row.get('work_industry'),
        university_id=row.get('university_id'),
        course=row.get('course'),
        sexual_orientation=row.get('sexual_orientation'),
        pets=row.get('pets'),
        activity_hours=str(row.get('activity_hours', '')),
        smoking=row.get('smoking'),
        extrovert_level=int(row.get('extrovert_level', 0)),
        cleanliness_level=int(row.get('cleanliness_level', 0)),
        partying_level=int(row.get('partying_level', 0)),
        sex_living_preference=str(row.get('sex_living_preference', '')),
        rent_location_preference=row.get('rent_location_preference'),
        age_preference=age_preference,
        rent_budget=rent_budget,
        last_filter_processed_at=None,
        available_at=row.get('available_at'),
        roommate_count_preference=row.get('roommate_count_preference'),
        interests=row.get('interests')
    )

def main():
    # Load data
    profiles_df, likes_df = load_data()
    print(f"Loaded {len(profiles_df)} profiles and {len(likes_df)} likes")
    
    # Create profile objects dictionary
    profile_dict = {}
    for _, row in profiles_df.iterrows():
        try:
            profile = create_profile(row)
            profile_dict[profile.user_id] = profile
        except Exception as e:
            print(f"Error creating profile for user {row['user_id']}: {e}")
    
    # Create training data using only actual likes
    training_data = []
    for _, row in likes_df.iterrows():
        viewer_id = row['user_id']
        liked_id = row['liked_user_id']
        
        viewer = profile_dict.get(viewer_id)
        candidate = profile_dict.get(liked_id)
        
        if viewer and candidate:
            training_data.append((viewer, candidate, True))
    
    print(f"Created {len(training_data)} training samples from actual likes")
    
    # Initialize and train the model
    recommender = XGBoostRecommender()
    recommender.train(training_data)
    
    # Save the trained model
    recommender.save_model('trained_xgboost_model.json')
    print("Model trained and saved successfully!")

if __name__ == "__main__":
    main()
