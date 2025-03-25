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
        id: int,
        created_at: datetime,
        contract_length: Optional[str],
        user_id: int,
        first_name: str,
        last_name: str,
        birth_date: date,
        is_verified: bool,
        gender: str,
        languages: Optional[List[str]],
        origin_country: str,
        occupation: str,
        work_industry: Optional[str],
        university_id: Optional[str],
        course_id: Optional[str],
        sexual_orientation: Optional[str],
        pets: Optional[str],
        activity_hours: str,
        smoking: Optional[str],
        extrovert_level: int,
        cleanliness_level: int,
        partying_level: int,
        sex_living_preference: str,
        age_preference: Tuple[int, int],
        rent_budget: Optional[Tuple[int, int]],
        available_at: Optional[str],
        age_similarity_weight=0.3,
        gender_similarity_weight=0.15,
        occupation_weight=0.125,
        special_origin_country_weight=0.2,
        university_weight=0.1,
        budget_weight=0,
        course_weight=0.05,
        work_industry_weight=0.05,
        smoking_weight=0,
        origin_country_weight=0.025,
        activity_hours_weight=0,
        likes=None,
        matches=None,
    ):
        self.user_id = user_id
        self.first_name = first_name
        self.last_name = last_name
        self.birth_date = birth_date
        self.is_verified = is_verified
        self.gender = gender
        self.languages = languages
        self.origin_country = origin_country
        self.occupation = occupation
        self.work_industry = work_industry
        self.university_id = university_id
        self.course_id = course_id
        self.contract_length = contract_length
        self.id = id
        self.created_at = created_at
        self.sexual_orientation = sexual_orientation
        self.pets = pets
        self.activity_hours = activity_hours
        self.smoking = smoking
        self.extrovert_level = extrovert_level
        self.cleanliness_level = cleanliness_level
        self.partying_level = partying_level
        self.sex_living_preference = sex_living_preference
        self.age_preference = age_preference
        self.rent_budget = rent_budget
        self.available_at = available_at
        # Weights
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

        # Initialize mutable objects properly
        self.likes = [] if likes is None else likes
        self.matches = [] if matches is None else matches
