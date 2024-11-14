import pandas as pd


class Profile:
    def __init__(self, user_id: int, age: int, gender: str, gender_preference: str, origin: str,
                 study_work: str, university_industry: str, budget_min: int, budget_max: int,
                 location: str, move_in_date: pd.Timestamp, roommates_wanted: int, smoker: bool,
                 lifestyle: str, interests: str, likes: int, lobies: int):
        self.user_id = user_id
        self.age = age
        self.gender = gender
        self.gender_preference = gender_preference
        self.origin = origin
        self.study_work = study_work
        self.university_industry = university_industry
        self.budget_min = budget_min
        self.budget_max = budget_max
        self.location = location
        self.move_in_date = move_in_date
        self.roommates_wanted = roommates_wanted
        self.smoker = smoker
        self.lifestyle = lifestyle
        self.interests = interests
        self.likes = likes
        self.lobies = lobies