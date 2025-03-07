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
        languages: Optional[List[str]],
        origin_country: str,
        occupation: str,
        work_industry: Optional[str],
        university_id: Optional[int],
        course_id: Optional[int],
        sexual_orientation: Optional[str],
        pets: Optional[str],
        activity_hours: str,
        smoking: Optional[str],
        extrovert_level: int,
        cleanliness_level: int,
        partying_level: int,
        available_at: Optional[str],
        id: Optional[int] = None,
        created_at: Optional[datetime] = None,
        contract_length: Optional[str] = None,
        rent_budget_range: Optional[Tuple[int, int]] = None,
        active_today: bool = False,
        preferred_gender: Optional[str] = None,
        age_range: Optional[Tuple[int, int]] = None,
        interests: Optional[List[str]] = None,
        likes: List = [],
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
        self.sexual_orientation = sexual_orientation
        self.pets = pets
        self.activity_hours = activity_hours
        self.smoking = smoking
        self.extrovert_level = extrovert_level
        self.cleanliness_level = cleanliness_level
        self.partying_level = partying_level
        self.available_at = available_at
        self.likes = likes
        self.id = id
        self.created_at = created_at
        self.contract_length = contract_length
        self.rent_budget_range = rent_budget_range
        self.active_today = active_today
        self.preferred_gender = preferred_gender
        self.age_range = age_range
        self.interests = interests if interests is not None else []


