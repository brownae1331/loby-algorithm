import os
import pandas as pd
from typing import List, Dict, Set
from datetime import datetime, timedelta, timezone
from generate_profiles import Profile
from helper_functions import calculate_age, CalculateScoreFunctions

def load_profile_matches(csv_path: str, profiles: List[Profile], days: int = 7) -> List[Profile]:
    """
    Load profile matches from CSV file and assign to profiles.
    
    Args:
        csv_path: Path to the profile_match CSV file
        profiles: List of Profile objects
        days: Number of days to consider for recent matches (default: 7)
        
    Returns:
        List of updated Profile objects with matches assigned
    """
    # Create a mapping of profile id to profile object for faster lookup
    profile_id_to_obj = {profile.id: profile for profile in profiles}
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Convert created_at to datetime with UTC timezone
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        
        # Calculate the cutoff date (7 days ago from now) with timezone info
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
        
        # Filter for matches within the specified time period
        recent_matches = df[df['created_at'] >= cutoff_date]
        
        print(f"Found {len(recent_matches)} matches from the last {days} days out of {len(df)} total matches")
        
        match_count = 0
        # Store match timestamps for later analysis
        match_timestamps = {}
        
        # Process each match
        for _, row in recent_matches.iterrows():
            profile_id_1 = row['profile_id_1']
            profile_id_2 = row['profile_id_2']
            created_at = row['created_at']
            
            # Record match timestamps for group recommendations
            match_key = (min(profile_id_1, profile_id_2), max(profile_id_1, profile_id_2))
            match_timestamps[match_key] = created_at
            
            # Add profile_id_2 to profile_id_1's matches
            if profile_id_1 in profile_id_to_obj:
                if profile_id_2 not in profile_id_to_obj[profile_id_1].matches:
                    profile_id_to_obj[profile_id_1].matches.append(profile_id_2)
                    match_count += 1
            
            # Add profile_id_1 to profile_id_2's matches
            if profile_id_2 in profile_id_to_obj:
                if profile_id_1 not in profile_id_to_obj[profile_id_2].matches:
                    profile_id_to_obj[profile_id_2].matches.append(profile_id_1)
                    match_count += 1
        
        print(f"Loaded {match_count} matches from the last {days} days")
        
        # Store match timestamps in a global variable for use in find_group_chat_recommendations
        global match_creation_times
        match_creation_times = match_timestamps
        
    except Exception as e:
        print(f"Error loading profile matches: {e}")
        # Print more detailed error information to help debug
        import traceback
        traceback.print_exc()
    
    return profiles

def check_hard_filters_compatibility(profile_a: Profile, profile_c: Profile) -> bool:
    """
    Check if two profiles pass each other's hard filters.
    Based on the criteria in assign_profiles_to_profile_list.
    
    Args:
        profile_a: First profile to check
        profile_c: Second profile to check
        
    Returns:
        Boolean indicating if profiles pass each other's hard filters
    """
    # Parse available_at dates
    def parse_date(date_str):
        if isinstance(date_str, str) and date_str == "IMMEDIATELY":
            return pd.Timestamp.now()
        try:
            return pd.to_datetime(date_str)
        except (ValueError, TypeError):
            return pd.Timestamp.now()
            
    profile_a_available_at = parse_date(profile_a.available_at)
    profile_c_available_at = parse_date(profile_c.available_at)
    
    # 1. Date range check (within 2 weeks)
    if not ((profile_a_available_at - timedelta(days=14)) <= profile_c_available_at <= 
            (profile_a_available_at + timedelta(days=14))):
        return False
    
    # 2. Gender preference check
    if not ((profile_a.sex_living_preference == "ANY" or profile_a.sex_living_preference == profile_c.gender) and
            (profile_c.sex_living_preference == "ANY" or profile_c.sex_living_preference == profile_a.gender)):
        return False
    
    # 3. Age preference check
    profile_a_age = calculate_age(profile_a.birth_date)
    profile_c_age = calculate_age(profile_c.birth_date)
    
    if not (profile_a.age_preference[0] <= profile_c_age <= profile_a.age_preference[1] and
            profile_c.age_preference[0] <= profile_a_age <= profile_c.age_preference[1]):
        return False
    
    # 4. Budget overlap check
    if not (CalculateScoreFunctions.calculate_budget_overlap_score(
        profile_a.rent_budget, profile_c.rent_budget) > 0):
        return False
    
    # All checks passed
    return True

def find_group_chat_recommendations(profiles: List[Profile]) -> List[Dict]:
    """
    Find potential group chat recommendations based on the following criteria:
    - User A and User B are matched
    - User B and User C are matched
    - User A and User C may not be matched but must pass each other's hard filters
    - Only consider matches from the last 7 days
    
    When Profile B forms a bridge between two compatible profiles (A and C), 
    Profile B will receive a recommendation to create a group chat.
    
    Args:
        profiles: List of Profile objects with matches
        
    Returns:
        List of dictionaries with group chat recommendations
    """
    # Create a mapping of profile id to profile object for faster lookup
    profile_id_to_obj = {profile.id: profile for profile in profiles}
    
    # Set to track unique groups (using frozenset to ensure uniqueness regardless of order)
    unique_groups = set()
    
    # List to store group recommendations
    group_recommendations = []
    
    # List to store bridge profiles that should receive recommendations
    bridge_profile_recommendations = []
    
    # Counters for reporting
    checked_trios = 0
    filtered_out = 0
    potential_bridges = 0
    
    # Get match timestamps from global variable
    global match_creation_times
    match_timestamps = globals().get('match_creation_times', {})
    
    # Check each profile's matches
    for profile_a in profiles:
        # For each match B of profile A
        for match_b_id in profile_a.matches:
            # Skip if match_b doesn't exist in our profiles
            if match_b_id not in profile_id_to_obj:
                continue
                
            profile_b = profile_id_to_obj[match_b_id]
            
            # For each match C of profile B
            for match_c_id in profile_b.matches:
                # Skip if match_c is the same as profile_a or doesn't exist
                if match_c_id == profile_a.id or match_c_id not in profile_id_to_obj:
                    continue
                    
                profile_c = profile_id_to_obj[match_c_id]
                checked_trios += 1
                
                # Check if profile_a and profile_c are already matched
                a_c_already_matched = profile_a.id in profile_c.matches
                
                # Check if they're compatible through hard filters if not already matched
                a_c_compatible = a_c_already_matched or check_hard_filters_compatibility(profile_a, profile_c)
                
                if not a_c_compatible:
                    # Not compatible
                    filtered_out += 1
                    continue
                
                # We found a valid group! Create a unique identifier
                group_ids = frozenset([profile_a.id, profile_b.id, profile_c.id])
                
                # Only add if we haven't seen this group before
                if group_ids not in unique_groups:
                    unique_groups.add(group_ids)
                    
                    # Identify which match happened more recently to determine if this is a new bridge
                    if match_timestamps:
                        ab_key = (min(profile_a.id, profile_b.id), max(profile_a.id, profile_b.id))
                        bc_key = (min(profile_b.id, profile_c.id), max(profile_b.id, profile_c.id))
                        
                        ab_time = match_timestamps.get(ab_key)
                        bc_time = match_timestamps.get(bc_key)
                        
                        # If both timestamps exist and profile A and C are not matched,
                        # we can create a bridge recommendation
                        if ab_time and bc_time and not a_c_already_matched:
                            potential_bridges += 1
                            
                            # Profile B should get a recommendation if it was their most recent match
                            # that completed the potential triangle
                            latest_match_time = max(ab_time, bc_time)
                            bridge_profile = profile_b
                            
                            bridge_profile_recommendations.append({
                                'bridge_profile': {
                                    'id': bridge_profile.id,
                                    'user_id': bridge_profile.user_id,
                                    'name': f"{bridge_profile.first_name} {bridge_profile.last_name}"
                                },
                                'other_profiles': [
                                    {
                                        'id': profile_a.id,
                                        'user_id': profile_a.user_id,
                                        'name': f"{profile_a.first_name} {profile_a.last_name}"
                                    },
                                    {
                                        'id': profile_c.id,
                                        'user_id': profile_c.user_id,
                                        'name': f"{profile_c.first_name} {profile_c.last_name}"
                                    }
                                ],
                                'most_recent_match_time': latest_match_time
                            })
                    
                    # Add to recommendations
                    group_recommendations.append({
                        'profiles': [
                            {
                                'id': profile_a.id,
                                'user_id': profile_a.user_id,
                                'name': f"{profile_a.first_name} {profile_a.last_name}"
                            },
                            {
                                'id': profile_b.id,
                                'user_id': profile_b.user_id,
                                'name': f"{profile_b.first_name} {profile_b.last_name}"
                            },
                            {
                                'id': profile_c.id,
                                'user_id': profile_c.user_id,
                                'name': f"{profile_c.first_name} {profile_c.last_name}"
                            }
                        ],
                        'already_matched': a_c_already_matched  # Indicates if all three are already matched
                    })
    
    print(f"Evaluated {checked_trios} potential trios")
    print(f"Filtered out {filtered_out} trios where A and C weren't compatible")
    print(f"Found {len(group_recommendations)} potential group chat recommendations")
    print(f"Found {potential_bridges} bridge profiles that can create groups")
    
    # Count how many groups have all three profiles already matched
    fully_matched_groups = sum(1 for g in group_recommendations if g['already_matched'])
    print(f"Groups where all three are already matched: {fully_matched_groups}")
    print(f"Groups where A and C aren't matched but are compatible: {len(group_recommendations) - fully_matched_groups}")
    
    # Count groups by profile (like SQL query results)
    profile_groups = {}
    for group in group_recommendations:
        profile_id = group['profiles'][0]['id']
        if profile_id not in profile_groups:
            profile_groups[profile_id] = 0
        profile_groups[profile_id] += 1

    if profile_groups:
        print("\nGroups by profile (like SQL query results):")
        for profile_id, group_count in profile_groups.items():
            profile_name = profile_id_to_obj[profile_id].first_name + " " + profile_id_to_obj[profile_id].last_name
            print(f"Profile {profile_id} ({profile_name}): {group_count} groups")
    
    # Display bridge profile recommendations
    if bridge_profile_recommendations:
        print("\nBridge Profile Recommendations:")
        # Sort by most recent match time (newest first)
        bridge_profile_recommendations.sort(key=lambda x: x['most_recent_match_time'], reverse=True)
        
        for i, rec in enumerate(bridge_profile_recommendations, 1):
            bridge_profile = rec['bridge_profile']
            other_profiles = rec['other_profiles']
            match_time = rec['most_recent_match_time']
            
            print(f"\nRecommendation {i}:")
            print(f"  Bridge Profile: {bridge_profile['name']} (ID: {bridge_profile['id']})")
            print(f"  Should create a group with:")
            for p in other_profiles:
                print(f"    - {p['name']} (ID: {p['id']})")
            print(f"  Based on match from: {match_time}")
    
    return group_recommendations
