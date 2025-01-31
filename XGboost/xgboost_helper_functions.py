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
        if smoking is None:
            return 0
        smoking_categories = ["NO", "YES", "SMOKE_WHAT"]
        if smoking not in smoking_categories:
            return 0
        return (
            smoking_categories.index(smoking) + 1
        )  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_origin_country_code(country: str) -> int:
        if country is None:
            return 0
        country_codes = [
            "AF",
            "AL",
            "DZ",
            "AD",
            "AO",
            "AG",
            "AR",
            "AM",
            "AU",
            "AT",
            "AZ",
            "BS",
            "BH",
            "BD",
            "BB",
            "BY",
            "BE",
            "BZ",
            "BJ",
            "BT",
            "BO",
            "BA",
            "BW",
            "BR",
            "BN",
            "BG",
            "BF",
            "BI",
            "KH",
            "CM",
            "CA",
            "CV",
            "CF",
            "TD",
            "CL",
            "CN",
            "CO",
            "KM",
            "CG",
            "CD",
            "CR",
            "CI",
            "HR",
            "CU",
            "CY",
            "CZ",
            "DK",
            "DJ",
            "DM",
            "DO",
            "EC",
            "EG",
            "SV",
            "GQ",
            "ER",
            "EE",
            "SZ",
            "ET",
            "FJ",
            "FI",
            "FR",
            "GA",
            "GM",
            "GE",
            "DE",
            "GH",
            "GR",
            "GD",
            "GT",
            "GN",
            "GW",
            "GY",
            "HT",
            "HN",
            "HU",
            "IS",
            "IN",
            "ID",
            "IR",
            "IQ",
            "IE",
            "IL",
            "IT",
            "JM",
            "JP",
            "JO",
            "KZ",
            "KE",
            "KI",
            "KP",
            "KR",
            "KW",
            "KG",
            "LA",
            "LV",
            "LB",
            "LS",
            "LR",
            "LY",
            "LI",
            "LT",
            "LU",
            "MG",
            "MW",
            "MY",
            "MV",
            "ML",
            "MT",
            "MH",
            "MR",
            "MU",
            "MX",
            "FM",
            "MD",
            "MC",
            "MN",
            "ME",
            "MA",
            "MZ",
            "MM",
            "NA",
            "NR",
            "NP",
            "NL",
            "NZ",
            "NI",
            "NE",
            "NG",
            "NO",
            "OM",
            "PK",
            "PW",
            "PA",
            "PG",
            "PY",
            "PE",
            "PH",
            "PL",
            "PT",
            "QA",
            "RO",
            "RU",
            "RW",
            "KN",
            "LC",
            "VC",
            "WS",
            "SM",
            "ST",
            "SA",
            "SN",
            "RS",
            "SC",
            "SL",
            "SG",
            "SK",
            "SI",
            "SB",
            "SO",
            "ZA",
            "SS",
            "ES",
            "LK",
            "SD",
            "SR",
            "SE",
            "CH",
            "SY",
            "TW",
            "TJ",
            "TZ",
            "TH",
            "TL",
            "TG",
            "TO",
            "TT",
            "TN",
            "TR",
            "TM",
            "TV",
            "UG",
            "UA",
            "AE",
            "GB",
            "US",
            "UY",
            "UZ",
            "VU",
            "VA",
            "VE",
            "VN",
            "YE",
            "ZM",
            "ZW",
        ]
        if country not in country_codes:
            return 0
        return (
            country_codes.index(country) + 1
        )  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_course_code(course_id: str) -> int:
        """Convert course_id to numeric code, handling None values"""
        if course_id is None:
            return 0  # Return 0 for no course
        try:
            return int(course_id)
        except (ValueError, TypeError):
            return 0  # Return 0 for invalid course IDs

    @staticmethod
    def get_occupation_code(occupation: str) -> int:
        if occupation is None:
            return 0
        occupation_categories = ["EMPLOYED", "CRUISING", "STUDENT"]
        if occupation not in occupation_categories:
            return 0
        return (
            occupation_categories.index(occupation) + 1
        )  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_work_industry_code(industry: str) -> int:
        if industry is None:
            return 0
        industry_categories = [
            "Agriculture",
            "Construction",
            "Creative Arts",
            "Education",
            "Finance",
            "Healthcare",
            "Hospitality",
            "IT",
            "Law",
            "Logistics",
            "Manufacturing",
            "Marketing",
            "Media",
            "Military",
            "Public Service",
            "Real Estate",
            "Recruitment",
            "Retail",
            "Social Care",
        ]
        if industry not in industry_categories:
            return 0
        return (
            industry_categories.index(industry) + 1
        )  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_activity_hours_code(activity_hours: str) -> int:
        if activity_hours is None:
            return 0
        activity_categories = ["NIGHT_OWL", "EARLY_BIRD"]
        if activity_hours not in activity_categories:
            return 0
        return (
            activity_categories.index(activity_hours) + 1
        )  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_university_id_code(university_id: str) -> int:
        if university_id is None:
            return 0
        try:
            uni_num = int(university_id)
            if 1 <= uni_num <= 250:
                return uni_num
            return 0  # Return 0 for out of range values
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def get_gender_code(gender: str) -> int:
        if gender is None:
            return 0
        gender_categories = ["MALE", "FEMALE"]
        if gender not in gender_categories:
            return 0
        return (
            gender_categories.index(gender) + 1
        )  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_sexual_orientation_code(orientation: str) -> int:
        if orientation is None:
            return 0
        orientation_categories = ["STRAIGHT", "PREFER_NOT_TO_SAY", "GAY", "BISEXUAL"]
        if orientation not in orientation_categories:
            return 0
        return (
            orientation_categories.index(orientation) + 1
        )  # Return index + 1 so 0 is reserved for unknown

    @staticmethod
    def get_extrovert_level_code(extrovert_level: str) -> int:
        if extrovert_level is None:
            return 0
        try:
            level = int(extrovert_level)
            if 1 <= level <= 5:
                return level
            return 0  # Return 0 for out of range values
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def get_cleanliness_level_code(cleanliness_level: str) -> int:
        if cleanliness_level is None:
            return 0
        try:
            level = int(cleanliness_level)
            if 1 <= level <= 5:
                return level
            return 0  # Return 0 for out of range values
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def get_partying_level_code(partying_level: str) -> int:
        if partying_level is None:
            return 0
        try:
            level = int(partying_level)
            if 1 <= level <= 5:
                return level
            return 0  # Return 0 for out of range values
        except (ValueError, TypeError):
            return 0

    @staticmethod
    def get_language_code(language: str) -> int:
        if language is None:
            return 0
        language_categories = [
            "English",
            "Romanian",
            "Russian",
            "Urdu",
            "Arabic",
            "Hindi",
            "Spanish",
            "Italian",
            "Indonesian",
            "Korean",
            "Tamil",
            "Telugu",
            "Kannada",
            "Croatian",
            "French",
            "Punjabi",
            "Somali",
        ]
        if language not in language_categories:
            return 0
        return (
            language_categories.index(language) + 1
        )  # Return index + 1 so 0 is reserved for unknown
