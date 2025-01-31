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

        self.likes = likes
