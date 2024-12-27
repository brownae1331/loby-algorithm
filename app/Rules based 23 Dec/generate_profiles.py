from datetime import date, datetime
from typing import Optional, List, Tuple

class Profile:
    def __init__(self, user_id: int, first_name: str, last_name: str, birth_date: date, is_verified: bool,
                 gender: str, description: Optional[str], languages: Optional[List[str]], origin_country: str,
                 occupation: str, work_industry: Optional[str], university_id: Optional[int], course: Optional[str],
                 sexual_orientation: Optional[str], pets: Optional[str], activity_hours: str, smoking: Optional[str],
                 extrovert_level: int, cleanliness_level: int, partying_level: int, sex_living_preference: str,
                 rent_location_preference: Optional[str], age_preference: Tuple[int, int], rent_budget: Optional[Tuple[int, int]],
                 last_filter_processed_at: Optional[datetime], available_at: Optional[str], roommate_count_preference: Optional[int],
                 interests: Optional[List[str]], budget_weight =  0.176, age_similarity_weight = 0.3, origin_country_weight = 0.029, course_weight = 0.169,
                 occupation_weight = 0.273, work_industry_weight = 0.094, smoking_weight = 0.03, activity_hours_weight = 0.017, available_at_weight = 0, university_weight = 0.2, likes = []):
        self.budget_weight = budget_weight
        self.age_similarity_weight = age_similarity_weight
        self.origin_country_weight = origin_country_weight
        self.course_weight = course_weight
        self.occupation_weight = occupation_weight
        self.work_industry_weight = work_industry_weight
        self.smoking_weight = smoking_weight
        self.activity_hours_weight = activity_hours_weight
        self.available_at_weight = available_at_weight
        self.university_weight = university_weight

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

        #implement the weights in the class

        self.likes = likes
    