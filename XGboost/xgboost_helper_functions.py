# origin_country
# course
# occupation
# work_industry
# smoking
# activity hours
# university_id

class FeatureEncoder:
    @staticmethod
    def get_smoking_code(smoking: str) -> int:
        smoking_categories = ["NO", "YES", "SMOKE_WHAT"]
        if smoking not in smoking_categories:
            return 0
        return smoking_categories.index(smoking) + 1  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_origin_country_code(country: str) -> int:
        country_codes = [
            "AF", "AL", "DZ", "AD", "AO", "AG", "AR", "AM", "AU", "AT", "AZ",
            "BS", "BH", "BD", "BB", "BY", "BE", "BZ", "BJ", "BT", "BO", "BA", 
            "BW", "BR", "BN", "BG", "BF", "BI",
            "KH", "CM", "CA", "CV", "CF", "TD", "CL", "CN", "CO", "KM", "CG", 
            "CD", "CR", "CI", "HR", "CU", "CY", "CZ",
            "DK", "DJ", "DM", "DO",
            "EC", "EG", "SV", "GQ", "ER", "EE", "SZ", "ET",
            "FJ", "FI", "FR",
            "GA", "GM", "GE", "DE", "GH", "GR", "GD", "GT", "GN", "GW", "GY",
            "HT", "HN", "HU",
            "IS", "IN", "ID", "IR", "IQ", "IE", "IL", "IT",
            "JM", "JP", "JO",
            "KZ", "KE", "KI", "KP", "KR", "KW", "KG",
            "LA", "LV", "LB", "LS", "LR", "LY", "LI", "LT", "LU",
            "MG", "MW", "MY", "MV", "ML", "MT", "MH", "MR", "MU", "MX", "FM", 
            "MD", "MC", "MN", "ME", "MA", "MZ", "MM",
            "NA", "NR", "NP", "NL", "NZ", "NI", "NE", "NG", "NO", "OM",
            "PK", "PW", "PA", "PG", "PY", "PE", "PH", "PL", "PT", "QA",
            "RO", "RU", "RW",
            "KN", "LC", "VC", "WS", "SM", "ST", "SA", "SN", "RS", "SC", "SL", 
            "SG", "SK", "SI", "SB", "SO", "ZA", "SS", "ES", "LK", "SD", "SR", 
            "SE", "CH", "SY",
            "TW", "TJ", "TZ", "TH", "TL", "TG", "TO", "TT", "TN", "TR", "TM", 
            "TV",
            "UG", "UA", "AE", "GB", "US", "UY", "UZ",
            "VU", "VA", "VE", "VN",
            "YE", "ZM", "ZW"
        ]
        if country not in country_codes:
            return 0
        return country_codes.index(country) + 1  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_course_code(course_id: str) -> int:
        course_num = int(course_id)
        if 1 <= course_num <= 250:
            return course_num
        return 0  # Return 0 for out of range values

    @staticmethod
    def get_occupation_code(occupation: str) -> int:
        occupation_categories = ["EMPLOYED", "CRUISING", "STUDENT"]
        if occupation not in occupation_categories:
            return 0
        return occupation_categories.index(occupation) + 1  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_work_industry_code(industry: str) -> int:
        industry_categories = [
            "Agriculture", "Construction", "Creative Arts", "Education", "Finance",
            "Healthcare", "Hospitality", "IT", "Law", "Logistics", "Manufacturing",
            "Marketing", "Media", "Military", "Public Service", "Real Estate",
            "Recruitment", "Retail", "Social Care"
        ]
        if industry not in industry_categories:
            return 0
        return industry_categories.index(industry) + 1  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_activity_hours_code(activity_hours: str) -> int:
        activity_categories = ["NIGHT_OWL", "EARLY_BIRD"]
        if activity_hours not in activity_categories:
            return 0
        return activity_categories.index(activity_hours) + 1  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_university_id_code(university_id: str) -> int:
        uni_num = int(university_id)
        if 1 <= uni_num <= 250:
            return uni_num
        return 0  # Return 0 for out of range values

    @staticmethod
    def get_gender_code(gender: str) -> int:
        gender_categories = ["MALE", "FEMALE"]
        if gender not in gender_categories:
            return 0
        return gender_categories.index(gender) + 1  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_sexual_orientation_code(orientation: str) -> int:
        orientation_categories = ["STRAIGHT", "PREFER_NOT_TO_SAY", "GAY", "BISEXUAL"]
        if orientation not in orientation_categories:
            return 0
        return orientation_categories.index(orientation) + 1  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_extrovert_level_code(extrovert_level: str) -> int:
        level = int(extrovert_level)
        if 1 <= level <= 5:
            return level
        return 0  # Return 0 for out of range values

    @staticmethod
    def get_cleanliness_level_code(cleanliness_level: str) -> int:
        level = int(cleanliness_level)
        if 1 <= level <= 5:
            return level
        return 0  # Return 0 for out of range values

    @staticmethod
    def get_partying_level_code(partying_level: str) -> int:
        level = int(partying_level)
        if 1 <= level <= 5:
            return level
        return 0  # Return 0 for out of range values