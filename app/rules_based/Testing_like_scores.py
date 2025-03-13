import os
import sys
import pandas as pd
import random

# Add the project root to Python path when running directly
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    sys.path.append(project_root)

from app.rules_based.helper_functions import (
    PrintFunctions,
    Constants,
    initialize_profile_list_from_csv,
    modify_weights_with_weighted_average,
    assign_profiles_to_profile_list,
    calculate_overall_score,
    calculate_age,
)


def import_liked_profiles(profiles_liked_csv, all_profiles):
    """Import and process profiles that liked each other from CSV, excluding opposite gender likes"""
    try:
        liked_df = pd.read_csv(profiles_liked_csv)
        # Convert date string to datetime
        liked_df["created_at"] = pd.to_datetime(liked_df["created_at"])
        print(
            "\nCompatibility Scores for Profiles that Liked Each Other (Same Gender Only):"
        )
        print("-" * 70)

        liked_pairs = []
        filtered_count = 0
        total_pairs = 0

        for _, row in liked_df.iterrows():
            total_pairs += 1
            user_1 = next(
                (p for p in all_profiles if p.id == row["profile_id_1"]), None
            )
            user_2 = next(
                (p for p in all_profiles if p.id == row["profile_id_2"]), None
            )

            if user_1 and user_2:
                # Skip if the users are of opposite genders
                # if user_1.gender != user_2.gender:
                #     filtered_count += 1
                #     continue

                # Calculate scores once
                score_1_to_2 = calculate_overall_score(user_1, user_2, row["location_score"])
                score_2_to_1 = calculate_overall_score(user_2, user_1, row["location_score"])
                avg_score = (score_1_to_2 + score_2_to_1) / 2

                # Include the date, location_score, and calculated scores in the liked_pairs
                liked_pairs.append((user_1, user_2, row["created_at"], row["location_score"], 
                                   score_1_to_2, score_2_to_1, avg_score))

                print(
                    f"Match: {user_1.first_name} {user_1.last_name} (ID: {user_1.id}) and "
                    f"{user_2.first_name} {user_2.last_name} (ID: {user_2.id})"
                )
                print(f"Gender: {user_1.gender} - {user_2.gender}")
                print(
                    f"Score {user_1.first_name} → {user_2.first_name}: {score_1_to_2:.2f}"
                )
                print(
                    f"Score {user_2.first_name} → {user_1.first_name}: {score_2_to_1:.2f}"
                )
                print(f"Location Score: {row['location_score']:.2f}")
                print(f"Final Score (with location): {avg_score:.2f}")
                print("-" * 70)

        print(f"\nTotal pairs processed: {total_pairs}")
        print(
            f"Pairs filtered (opposite gender or outside basic filters): {filtered_count}"
        )
        print(f"Remaining pairs: {len(liked_pairs)}")

        return liked_pairs
    except FileNotFoundError:
        print(
            f"\nWarning: {profiles_liked_csv} not found. Starting with empty liked profiles."
        )
        return []
    except Exception as e:
        print(f"\nError reading profiles_liked.csv: {str(e)}")
        return []


def run():
    # Initialize profiles from CSV
    csv_path = os.path.join(
        os.path.dirname(__file__),
        "Profiles_11-03.csv",
    )
    profiles_liked_csv = os.path.join(
        os.path.dirname(__file__), "likes_11-03.csv"
    )
    all_profiles = initialize_profile_list_from_csv(csv_path)

    # Import liked profiles and calculate scores
    liked_pairs = import_liked_profiles(profiles_liked_csv, all_profiles)

    # Calculate and display average scores by month
    print("\nSummary of Liked Pairs by Month:")
    print("-" * 70)

    # Group pairs by month
    monthly_stats = {}
    total_score = 0
    total_pairs = 0
    total_high_scores = 0

    for pair in liked_pairs:
        # Unpack all values including pre-calculated scores
        _, _, date, _, _, _, avg_score = pair

        month_key = date.strftime("%Y-%m")

        # Update overall statistics
        total_score += avg_score
        total_pairs += 1
        if avg_score > 0.7:
            total_high_scores += 1

        if month_key not in monthly_stats:
            monthly_stats[month_key] = {
                "total_score": 0,
                "pair_count": 0,
                "high_score_count": 0,
                "scores": [],
            }

        monthly_stats[month_key]["total_score"] += avg_score
        monthly_stats[month_key]["pair_count"] += 1
        monthly_stats[month_key]["scores"].append(avg_score)
        if avg_score > 0.7:
            monthly_stats[month_key]["high_score_count"] += 1

    # Print monthly statistics
    for month in sorted(monthly_stats.keys()):
        stats = monthly_stats[month]
        if stats["pair_count"] > 0:
            monthly_avg = stats["total_score"] / stats["pair_count"]
            monthly_median = pd.Series(stats["scores"]).median()  # Calculate median
            high_score_percentage = (
                stats["high_score_count"] / stats["pair_count"]
            ) * 100
            print(f"\nMonth: {month}")
            print(f"Overall average compatibility score: {monthly_avg:.2f}")
            print(f"Median compatibility score: {monthly_median:.2f}")
            print(f"Total number of liked pairs: {stats['pair_count']}")
            print(f"Percentage of likes with score >0.7: {high_score_percentage:.1f}%")
            print("-" * 70)

    # Print overall statistics
    if total_pairs > 0:
        overall_avg = total_score / total_pairs
        overall_high_score_percentage = (total_high_scores / total_pairs) * 100
        print("\nOverall Statistics Across All Months:")
        print("-" * 70)
        print(f"Overall average compatibility score: {overall_avg:.2f}")
        print(
            f"Percentage of all likes with score >0.7: {overall_high_score_percentage:.1f}%"
        )
        print(f"Total number of liked pairs: {total_pairs}")

    if not monthly_stats:
        print("\nNo liked pairs found.")


if __name__ == "__main__":
    run()
