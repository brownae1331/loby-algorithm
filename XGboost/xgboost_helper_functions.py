

origin_country
course
occupation
work_industry
smoking
activity hours
university_id


def get_origin_country_code(country: str) -> int:
    country_codes = [
        "UK", "USA", "Canada", "Australia"
    ]
    if country not in country_codes:
        return 0
    return country_codes.index(country)  # Return 0 if country not found

def get_course_code(course: str) -> int:
    course_codes = {"Computer Science", "Engineering", "Physics", "Chemistry", "Biology",
                    "History", "Literature", "Philosophy", "Art", "Music",
                    "Business", "Economics", "Finance", "Marketing"
    }
    return course_codes.get(course, 0)  # Return 0 if course not found

def get_occupation_code(occupation: str) -> int:
    occupation_codes = {
        "Student": 1,
        "Engineer": 2,
        "Doctor": 3,
        "Teacher": 4,
        "Artist": 5,
        # Add more occupations as needed
    }
    return occupation_codes.get(occupation, 0)  # Return 0 if occupation not found

def get_work_industry_code(industry: str) -> int:
    industry_codes = {
        "Technology": 1,
        "Healthcare": 2,
        "Education": 3,
        "Finance": 4,
        "Arts": 5,
        # Add more industries as needed
    }
    return industry_codes.get(industry, 0)  # Return 0 if industry not found

def get_smoking_code(smoking: str) -> int:
    smoking_codes = {
        "Non-smoker": 1,
        "Occasional smoker": 2,
        "Regular smoker": 3,
    }
    return smoking_codes.get(smoking, 0)  # Return 0 if smoking status not found

def get_activity_hours_code(activity_hours: str) -> int:
    activity_hours_codes = {
        "Low": 1,
        "Medium": 2,
        "High": 3,
    }
    return activity_hours_codes.get(activity_hours, 0)  # Return 0 if activity level not found

def get_university_id_code(university_id: str) -> int:
    university_codes = {
        "UCL": 1,
        "Oxford": 2,
        "Cambridge": 3,
        "Imperial": 4,
        "LSE": 5,
        # Add more universities as needed
    }
    return university_codes.get(university_id, 0)  # Return 0 if university not found