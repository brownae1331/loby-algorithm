import numpy as np
import pandas as pd
from typing import List, Union
from profile import Profile

# Ensure all arrays have the same length
num_profiles = 100

# Example profile data
profiles = pd.DataFrame({
    "user_id": range(1, num_profiles + 1),
    "age": np.random.randint(18, 35, num_profiles),
    "gender": np.random.choice(["M", "F"], num_profiles),
    "gender_preference": np.random.choice(["M", "F", "Both"], num_profiles),
    "from": np.random.choice(["London"], num_profiles),
    "study_work": np.random.choice(["Student", "Working"], num_profiles),
    "university_industry": np.random.choice(["UCL", "Tech", "Imperial", "Finance", "King's College", "Media"], num_profiles),
    "budget_min": np.random.randint(800, 1000, num_profiles),
    "budget_max": np.random.randint(1500, 3800, num_profiles),
    "location": np.random.choice(["NW", "SW", "NE", "SE"], num_profiles),
    "move_in_date": pd.to_datetime(np.random.choice(pd.date_range("2024-01-01", "2024-12-31"), num_profiles)),
    "roommates_wanted": np.random.randint(1, 4, num_profiles),
    "smoker": np.random.choice([True, False], num_profiles),
    "lifestyle": np.random.choice(["Night Owl", "Early Bird"], num_profiles),
    "interests": np.random.choice(["Music, Movies", "Hiking, Cooking", "Sports, Travel", "Reading, Fitness", "Art, Gaming"], num_profiles),
    "likes": np.random.randint(0, 10, num_profiles),
    "lobies": np.random.randint(0, 6, num_profiles)
})

# Convert DataFrame rows to Profile objects
profile_objects = [
    Profile(
        user_id=row["user_id"],
        age=row["age"],
        gender=row["gender"],
        gender_preference=row["gender_preference"],
        origin=row["from"],
        study_work=row["study_work"],
        university_industry=row["university_industry"],
        budget_min=row["budget_min"],
        budget_max=row["budget_max"],
        location=row["location"],
        move_in_date=row["move_in_date"],
        roommates_wanted=row["roommates_wanted"],
        smoker=row["smoker"],
        lifestyle=row["lifestyle"],
        interests=row["interests"],
        likes=row["likes"],
        lobies=row["lobies"]
    )
    for _, row in profiles.iterrows()
]

def assign_profiles_to_clusters(profile_ids: Union[int, List[int]] = None) -> None:
    """Assign profiles to clusters based on some criteria."""
    if profile_ids is None:
        selected_profiles = profiles
    else:
        if isinstance(profile_ids, int):
            profile_ids = [profile_ids]
        selected_profiles = profiles[profiles['user_id'].isin(profile_ids)]

    clusters = {}
    for _, profile in selected_profiles.iterrows():
        cluster_key = (profile['location'], profile['roommates_wanted'], profile['gender_preference'])
        if cluster_key not in clusters:
            clusters[cluster_key] = []
        clusters[cluster_key].append(profile['user_id'])

    for cluster_key, user_ids in clusters.items():
        print(f"Cluster {cluster_key}: {user_ids}")

# Example usage: Assign all profiles to clusters
assign_profiles_to_clusters()