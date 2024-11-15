import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional, List, Union
from profile import Profile
from compatibility import CompatibilityCalculator

# Ensure all arrays have the same length
num_profiles = 100

# Example profile data
profiles = pd.DataFrame({
    "user_id": range(1, num_profiles + 1),
    "first_name": np.random.choice(["John", "Jane", "Alice", "Bob"], num_profiles),
    "last_name": np.random.choice(["Doe", "Smith", "Johnson", "Williams"], num_profiles),
    "birth_date": pd.to_datetime(np.random.choice(pd.date_range("1980-01-01", "2005-12-31"), num_profiles)),
    "is_verified": np.random.choice([True, False], num_profiles),
    "gender": np.random.choice(["M", "F"], num_profiles),
    "description": np.random.choice([None, "Loves hiking", "Enjoys cooking", "Avid reader"], num_profiles),
    "languages": [np.random.choice(["English", "Spanish", "French", "Welsh"], np.random.randint(1, 4)).tolist() for _ in range(num_profiles)],
    "origin_country": np.random.choice(["UK", "USA", "Canada", "Australia"], num_profiles),
    "occupation": np.random.choice(["Student", "Working", "Cruising"], num_profiles),
    "work_industry": np.random.choice([None, "Tech", "Finance", "Media"], num_profiles),
    "university_id": np.random.choice([None, 1, 2, 3], num_profiles),
    "course": np.random.choice([None, "Computer Science", "Business", "Engineering"], num_profiles),
    "sexual_orientation": np.random.choice([None, "Heterosexual", "Homosexual", "Bisexual"], num_profiles),
    "pets": np.random.choice([None, "Dog", "Cat", "None"], num_profiles),
    "activity_hours": np.random.choice(["Morning", "Evening", "Night"], num_profiles),
    "smoking": np.random.choice([None, "Yes", "No"], num_profiles),
    "extrovert_level": np.random.randint(1, 10, num_profiles),
    "cleanliness_level": np.random.randint(1, 10, num_profiles),
    "partying_level": np.random.randint(1, 10, num_profiles),
    "sex_living_preference": np.random.choice(["Male", "Female", "Both"], num_profiles),
    "rent_location_preference": np.random.choice([None, "NW", "SW", "NE", "SE"], num_profiles),
    "age_preference": [(18, 25) for _ in range(num_profiles)],
    "rent_budget": [(800, 1500) for _ in range(num_profiles)],
    "last_filter_processed_at": pd.to_datetime(np.random.choice(pd.date_range("2023-01-01", "2023-12-31"), num_profiles)),
    "available_at": np.random.choice([None, "2024-01-01", "2024-06-01"], num_profiles),
    "roommate_count_preference": np.random.choice([1, 2, 3], num_profiles),
    "interests": [np.random.choice(["Reading", "Traveling", "Cooking", "Sports"], np.random.randint(1, 4)).tolist() for _ in range(num_profiles)]
})

# Convert DataFrame rows to Profile objects
profile_objects = [
    Profile(
        user_id=row["user_id"],
        first_name=row["first_name"],
        last_name=row["last_name"],
        birth_date=row["birth_date"],
        is_verified=row["is_verified"],
        gender=row["gender"],
        description=row["description"],
        languages=row["languages"],
        origin_country=row["origin_country"],
        occupation=row["occupation"],
        work_industry=row["work_industry"],
        university_id=row["university_id"],
        course=row["course"],
        sexual_orientation=row["sexual_orientation"],
        pets=row["pets"],
        activity_hours=row["activity_hours"],
        smoking=row["smoking"],
        extrovert_level=row["extrovert_level"],
        cleanliness_level=row["cleanliness_level"],
        partying_level=row["partying_level"],
        sex_living_preference=row["sex_living_preference"],
        rent_location_preference=row["rent_location_preference"],
        age_preference=row["age_preference"],
        rent_budget=row["rent_budget"],
        last_filter_processed_at=row["last_filter_processed_at"],
        available_at=row["available_at"],
        roommate_count_preference=row["roommate_count_preference"],
        interests=row["interests"]
    )
    for _, row in profiles.iterrows()
]

def calculate_age(birth_date):
    """Calculate age from birth date."""
    today = date.today()
    return today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))


# Define clusters as a global variable
clusters = {}

def assign_profiles_to_clusters(profile_ids: Union[int, List[int]] = None) -> None:
    """Assign profiles to clusters based on some criteria."""
    global clusters
    if profile_ids is None:
        selected_profiles = profiles
    else:
        if isinstance(profile_ids, int):
            profile_ids = [profile_ids]
        selected_profiles = profiles[profiles['user_id'].isin(profile_ids)]

    clusters = {}
    for _, profile in selected_profiles.iterrows():
        # Adjust clustering logic to handle sex_living_preference
        sex_pref = profile['sex_living_preference']
        if sex_pref == "Both":
            cluster_keys_list = [
                (profile['rent_location_preference'], profile['roommate_count_preference'], "Male"),
                (profile['rent_location_preference'], profile['roommate_count_preference'], "Female")
            ]
        else:
            cluster_keys_list = [(profile['rent_location_preference'], profile['roommate_count_preference'], sex_pref)]

        for cluster_key in cluster_keys_list:
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(profile['user_id'])

    for cluster_key, user_ids in clusters.items():
        print(f"Cluster {cluster_key}: {user_ids}")
        sex_prefs_in_cluster = set(profiles[profiles['user_id'].isin(user_ids)]['sex_living_preference'])
        genders_in_cluster = list(sex_prefs_in_cluster)[0]
        print(f"Sex living preferences in cluster {cluster_key}: {sex_prefs_in_cluster}")
        print(f"Genders in cluster {cluster_key}: {genders_in_cluster}")
        print(" ")


# Example usage: Assign all profiles to clusters
assign_profiles_to_clusters()

def score_profiles_in_cluster(cluster_key, user_ids):
    """Score profiles within a cluster."""
    if len(user_ids) <= 3:
        print(f"Cluster {cluster_key} does not have more than 3 users.")
        return

    for i in range(len(user_ids)):
        for j in range(i + 1, len(user_ids)):
            p1 = profile_objects[user_ids[i] - 1]
            p2 = profile_objects[user_ids[j] - 1]
            calculator = CompatibilityCalculator(p1, p2)
            score = calculator.calculate()
            print(f"Score between user {user_ids[i]} and user {user_ids[j]}: {score}")

# Find a cluster with more than 3 users
for cluster_key, user_ids in clusters.items():
    if len(user_ids) > 3:
        score_profiles_in_cluster(cluster_key, user_ids)
        break