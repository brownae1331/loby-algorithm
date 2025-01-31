import pickle
from app.rules_based.helper_functions import (
    initialize_profile_list,
    calculate_overall_score,
)


def generate_swipe_history(profiles):
    swipe_history = []
    total_profiles = len(profiles)
    for i, viewer_profile in enumerate(profiles):
        for j, swiped_profile in enumerate(profiles):
            if i != j:
                score = calculate_overall_score(viewer_profile, swiped_profile)
                liked = score >= 0.5  # Assuming a threshold of 0.8 for "like"
                swipe_history.append((viewer_profile, swiped_profile, liked))
            if (j + 1) % 100 == 0:
                print(f"Processed {i + 1}/{total_profiles} viewer profiles")
    return swipe_history


test_profiles = initialize_profile_list()
swipe_history = generate_swipe_history(test_profiles)

with open("swipe_history.pkl", "wb") as file:
    pickle.dump(swipe_history, file)
