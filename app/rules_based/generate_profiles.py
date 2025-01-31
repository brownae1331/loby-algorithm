# import os
# import sys
# import pandas as pd
# # Add the project root to Python path when running directly
# if __name__ == "__main__":
#     project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
#     sys.path.append(project_root)

from datetime import date, datetime
from typing import Optional, List, Tuple


class Profile:
    def __init__(
        self,
        user_id: int,
        first_name: str,
        last_name: str,
        birth_date: date,
        is_verified: bool,
        gender: str,
        description: Optional[str],
        languages: Optional[List[str]],
        origin_country: str,
        occupation: str,
        work_industry: Optional[str],
        university_id: Optional[str],
        course: Optional[str],
        sexual_orientation: Optional[str],
        pets: Optional[str],
        activity_hours: str,
        smoking: Optional[str],
        extrovert_level: int,
        cleanliness_level: int,
        partying_level: int,
        sex_living_preference: str,
        rent_location_preference: Optional[str],
        age_preference: Tuple[int, int],
        rent_budget: Optional[Tuple[int, int]],
        last_filter_processed_at: Optional[datetime],
        available_at: Optional[str],
        roommate_count_preference: Optional[int],
        interests: Optional[List[str]],
        age_similarity_weight=0.6,
        gender_similarity_weight=0.4,
        occupation_weight=0.273,
        special_origin_country_weight=0.2,
        university_weight=0.2,
        budget_weight=0.176,
        course_weight=0.169,
        work_industry_weight=0.094,
        smoking_weight=0.03,
        origin_country_weight=0.029,
        activity_hours_weight=0.017,
        likes=[],
    ):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.birth_date = birth_date
        self.is_verified = is_verified
        self.gender = gender
        self.description = description
        self.languages = languages
        self.origin_country = origin_country
        self.occupation = occupation
        self.work_industry = work_industry
        self.university_id = university_id
        self.course = course
        self.sexual_orientation = sexual_orientation
        self.pets = pets
        self.activity_hours = activity_hours
        self.smoking = smoking
        self.extrovert_level = extrovert_level
        self.cleanliness_level = cleanliness_level
        self.partying_level = partying_level
        self.sex_living_preference = sex_living_preference
        self.rent_location_preference = rent_location_preference
        self.age_preference = age_preference
        self.rent_budget = rent_budget
        self.last_filter_processed_at = last_filter_processed_at
        self.available_at = available_at
        self.roommate_count_preference = roommate_count_preference
        self.interests = interests

        # Weights for matching algorithm
        self.age_similarity_weight = age_similarity_weight
        self.gender_similarity_weight = gender_similarity_weight
        self.occupation_weight = occupation_weight
        self.origin_country_weight = origin_country_weight
        self.special_origin_country_weight = special_origin_country_weight
        self.university_weight = university_weight
        self.budget_weight = budget_weight
        self.course_weight = course_weight
        self.work_industry_weight = work_industry_weight
        self.smoking_weight = smoking_weight
        self.activity_hours_weight = activity_hours_weight

        self.likes = likes
