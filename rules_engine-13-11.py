import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from faker import Faker

fake = Faker()

# Define date ranges
start_date = "2024-11-13"
end_date = "2025-01-01"

# Generate 100 random profiles
random_data = {
    "user_id": range(1, 100),  # Start IDs after the existing 5 users
    "age": np.random.randint(18, 35, 100),
    "gender": np.random.choice(["M", "F"], 100),
    "from": np.random.choice(["London"], 100),
    "study_work": np.random.choice(["Student", "Working"], 100),
    "university_industry": np.random.choice(["UCL", "Tech", "Imperial", "Finance", "King's College", "Media"], 100),
    "budget_min": np.random.randint(800, 1000, 100),
    "budget_max": np.random.randint(1500, 3800, 100),
    "location": np.random.choice(["NW1", "SW3", "NE1", "SE7"], 100),
    "move_in_date": ("2024-01-01"),
    "roommates_wanted": np.random.randint(1, 4, 100),
    "smoker": np.random.choice([True, False], 100),
    "lifestyle": np.random.choice(["Night Owl", "Early Bird"], 100),
    "interests": np.random.choice(["Music, Movies", "Hiking, Cooking", "Sports, Travel", "Reading, Fitness", "Art, Gaming"], 100),
    "likes": np.random.randint(0, 10, 100),
    "lobies": np.random.randint(0, 6, 100)
}

print (random_data)