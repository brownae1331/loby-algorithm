from profile import Profile

class CompatibilityCalculator:
    """Calculator used to determine the compatibility score between two profiles."""

    def __init__(self, p1: Profile, p2: Profile):
        """Initialize a calculator with profiles and score trackers."""
        self.p1 = p1
        self.p2 = p2
        self.total_points: float = 0
        self.scored_points: float = 0

    def calculate(self) -> float:
        """Calculate and return how compatible the two profiles are."""
        self.compare_origin_country()
        self.compare_occupation_and_universities()
        self.compare_activity_hours()
        self.compare_smoking()
        self.compare_interests()
        self.compare_languages()
        self.compare_pets()
        self.compare_extrovert_level()
        self.compare_cleanliness_level()
        self.compare_partying_level()

        return round((self.scored_points * 100) / self.total_points, 2)

    def compare_origin_country(self) -> None:
        """Compare origin country between two profiles."""
        self.total_points += 1

        if self.p1.origin_country == self.p2.origin_country:
            self.scored_points += 1

    def compare_occupation_and_universities(self) -> None:
        """Compare occupation, industry, university and course between two profiles."""
        self.total_points += 1

        if self.p1.occupation == self.p2.occupation:
            self.scored_points += 1

        if (
            self.p1.occupation == Occupation.CRUISING
            or self.p1.occupation == Occupation.CRUISING
        ):
            return

        if self.p1.work_industry and self.p2.work_industry:
            self.total_points += 1

            if self.p1.work_industry == self.p2.work_industry:
                self.scored_points += 1

        if self.p1.university and self.p2.university:
            self.total_points += 1

            if self.p1.university == self.p2.university:
                self.scored_points += 1
        else:
            return

        if self.p1.course and self.p2.course:
            self.total_points += 1

            if self.p1.course == self.p2.course:
                self.scored_points += 1

    def compare_activity_hours(self) -> None:
        """Compare activity hours between two profiles."""
        self.total_points += 1

        if self.p1.activity_hours == self.p2.activity_hours:
            self.scored_points += 1

    def compare_smoking(self) -> None:
        """Compare smoking between two profiles."""
        if self.p1.smoking is None or self.p2.smoking is None:
            return

        self.total_points += 1

        if self.p1.smoking == self.p2.smoking:
            self.scored_points += 1

    def compare_interests(self) -> None:
        """Compare interests between two profiles."""
        self.total_points += len(self.p1.interests)

        p1_interests = [interest.interest.name for interest in self.p1.interests]
        p2_interests = [interest.interest.name for interest in self.p2.interests]
        self.scored_points += count_common_elements(p1_interests, p2_interests)

    def compare_languages(self) -> None:
        """Compare languages between two profiles."""
        if self.p1.languages is None or self.p2.languages is None:
            return

        self.total_points += 1

        has_common_language: bool = has_common_element(
            self.p1.languages_list, self.p2.languages_list
        )
        if has_common_language:
            self.scored_points += 1

    def compare_pets(self) -> None:
        """Compare pets between two profiles."""
        self.total_points += 1

        if self.p1.pets is None and self.p2.pets is None:
            self.scored_points += 1

        elif self.p1.pets and self.p2.pets:
            has_common_pet: bool = has_common_element(
                self.p1.pets_list, self.p2.pets_list
            )
            if has_common_pet:
                self.scored_points += 1

    def compare_extrovert_level(self) -> None:
        """Compare extrovert level between two profiles."""
        self.total_points += 1

        self.scored_points += self.__compare_level_type_fields(
            self.p1.extrovert_level, self.p2.extrovert_level
        )

    def compare_cleanliness_level(self) -> None:
        """Compare cleanliness level between two profiles."""
        self.total_points += 1

        self.scored_points += self.__compare_level_type_fields(
            self.p1.cleanliness_level, self.p2.cleanliness_level
        )

    def compare_partying_level(self) -> None:
        """Compare partying level between two profiles."""
        self.total_points += 1

        self.scored_points += self.__compare_level_type_fields(
            self.p1.partying_level, self.p2.partying_level
        )

    def __compare_level_type_fields(self, value1: int, value2: int) -> float:
        """Compare level type profile fields and get the number of scored points.

        Level type fields are: cleanliness, partying and extrovert.
        Points awarded:
            1 - levels are equal;
            0.5 - levels are 1 unit apart;
            0 - levels are more than 1 unit apart.
        """
        gap_distance = max(value1, value2) - min(value1, value2)
        if gap_distance == 0:
            return 1
        elif gap_distance == 1:
            return 0.5

        return 0
