import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Optional, List, Union

from matplotlib.style.core import available

from generate_profiles import Profile
from compatibility import CompatibilityCalculator

# Ensure all arrays have the same length
num_profiles = 500

# Generate a range of dates for the year 2024
date_range = pd.date_range("2024-06-01", "2024-12-31")


# Example profile data
profiles = pd.DataFrame({
    "user_id": range(1, num_profiles + 1),
    "first_name": np.random.choice(["John", "Jane", "Alice", "Bob"], num_profiles),
    "last_name": np.random.choice(["Doe", "Smith", "Johnson", "Williams"], num_profiles),
    "birth_date": pd.to_datetime(np.random.choice(pd.date_range("1995-01-01", "2006-12-31"), num_profiles)),
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
    "rent_location_preference": np.random.choice(["London", "Bath", "Leeds", "Oxford"], num_profiles),
    "age_preference": [(18, 25) for _ in range(num_profiles)],
    "rent_budget": [(np.random.randint(500, 700), np.random.randint(700, 1000)) for _ in range(num_profiles)],
    "last_filter_processed_at": pd.to_datetime(np.random.choice(pd.date_range("2023-01-01", "2023-12-31"), num_profiles)),
    "available_at": np.random.choice(date_range.strftime('%Y-%m'), num_profiles),
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

        location = profile['rent_location_preference']
        available_at = profile['available_at']

        cluster_keys = [(available_at, location)]

        for cluster_key in cluster_keys:
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            clusters[cluster_key].append(profile['user_id'])

    for cluster_key, user_ids in clusters.items():
        print(f"Cluster {cluster_key}: {user_ids}")
        budget_pref = set(profiles[profiles['user_id'].isin(user_ids)]['rent_budget'])
        genders_in_cluster = set(profiles[profiles['user_id'].isin(user_ids)]['gender'])
        print(f"budget {cluster_key}: {budget_pref}")
        print(" ")


# Example usage: Assign all profiles to clusters
assign_profiles_to_clusters()

def assign_sub_clusters(profile_ids: Union[int, List[int]] = None) -> None:
    """Assign profiles to sub-clusters based on age, gender, and budget."""
    global clusters, sub_clusters
    sub_clusters = {}

    if profile_ids is None:
        selected_profiles = profiles
    else:
        if isinstance(profile_ids, int):
            profile_ids = [profile_ids]
        selected_profiles = profiles[profiles['user_id'].isin(profile_ids)]

    for cluster_key, user_ids in clusters.items():
        sub_clusters[cluster_key] = {}
        for user_id in user_ids:
            profile = selected_profiles[selected_profiles['user_id'] == user_id].iloc[0]
            age = calculate_age(profile['birth_date'])
            gender = profile['gender']
            budget = profile['rent_budget'][0]  # Assuming rent_budget is a tuple (min, max)

            sub_cluster_key = (f"{age - 3}-{age + 3}", gender, f"{budget - 100}-{budget + 100}")
            if sub_cluster_key not in sub_clusters[cluster_key]:
                sub_clusters[cluster_key][sub_cluster_key] = []
            sub_clusters[cluster_key][sub_cluster_key].append(user_id)

    for cluster_key, sub_cluster in sub_clusters.items():
        print(f"Cluster {cluster_key}:")
        for sub_cluster_key, user_ids in sub_cluster.items():
            print(f"  Sub-cluster {sub_cluster_key}: {user_ids}")
        print(" ")


assign_sub_clusters()

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