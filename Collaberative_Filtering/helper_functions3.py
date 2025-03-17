import csv
import pandas as pd
from datetime import datetime, date
from typing import List, Tuple, Dict
from generate_profiles3 import Profile, Apartment, UserPropertyFilter

def parse_date(date_str):
    """Parse date string to date object."""
    if isinstance(date_str, str):
        if date_str == "IMMEDIATELY":
            return date.today()
        try:
            return datetime.strptime(date_str.split()[0], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None

def parse_range(range_str):
    """Parse range string like '[100,801)' to tuple (100, 801)."""
    if isinstance(range_str, str) and range_str.startswith('[') and ')' in range_str:
        nums = range_str.strip('[]()').split(',')
        return (int(nums[0]), int(nums[1]))
    return None

def load_profiles(csv_path):
    """Load profiles from CSV file."""
    profiles = []
    user_filters = {}
    
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        # Create Profile object
        profile = Profile(
            user_id=row['user_id'],
            first_name=row['first_name'],
            last_name=row['last_name'],
            birth_date=parse_date(row['birth_date']),
            is_verified=row['is_verified'] == 'true',
            gender=row['gender'],
            languages=None,  # Not in CSV
            origin_country=row['origin_country'],
            occupation=row['occupation'],
            work_industry=row['work_industry'] if row['work_industry'] != '' else None,
            university_id=int(row['university_id']) if row['university_id'] and row['university_id'] != '' else None,
            course_id=int(row['course_id']) if row['course_id'] and row['course_id'] != '' else None,
            sexual_orientation=None,  # Not in CSV
            pets=None,  # Not in CSV
            activity_hours=row['activity_hours'],
            smoking=row['smoking'],
            extrovert_level=0,  # Not in CSV
            cleanliness_level=0,  # Not in CSV
            partying_level=0,  # Not in CSV
            available_at=row['available_at'],
            id=row['profile_id'],
            created_at=datetime.strptime(row['profile_created_at'].split('+')[0].strip(), '%Y-%m-%d %H:%M:%S.%f'),
            contract_length=row['contract_length'],
            rent_budget_range=parse_range(row['profile_rent_budget']),
            active_today=False,  # Not in CSV
            preferred_gender=row['preferred_gender'],
            age_range=parse_range(row['age_range']),
            interests=None,  # Not in CSV
            likes=[]  # Will fill this later from enquiries
        )
        profiles.append(profile)
        
        # Create UserPropertyFilter object
        filter_created_at = datetime.strptime(row['property_filter_created_at'].split('+')[0].strip(), '%Y-%m-%d %H:%M:%S.%f')
        user_apt_filters = UserPropertyFilter(
            user_id=row['user_id'],
            apt_price_range=parse_range(row['property_rent_budget']),
            bedroom_count_range=parse_range(row['bedroom_count_range']),
            property_type=row['property_type'],
            created_at=filter_created_at
        )
        user_apt_filters[row['user_id']] = user_apt_filters
    
    return profiles, user_apt_filters

def load_apartments(csv_path):
    """Load apartments from CSV file."""
    apartments = []
    
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        # Create Apartment object
        apartment = Apartment(
            id=row['id'],
            created_at=datetime.strptime(row['created_at'].split('+')[0].strip(), '%Y-%m-%d %H:%M:%S.%f'),
            uuid=int(hash(row['uuid']) % 1000000),  # Convert UUID to int
            address_line_1=row['address_line_1'],
            address_line_2=row['address_line_2'] if row['address_line_2'] != '' else None,
            city=row['city'],
            postcode=row['postcode'],
            display_address=row['display_address'],
            description=row['description'],
            short_description=row['short_description'],
            bedroom_count=row['bedroom_count'],
            bathroom_count=row['bathroom_count'],
            reception_count=row['reception_count'],
            property_type=row['property_type'],
            available_at=parse_date(row['available_at']),
            cost=row['cost'],
            contract_length=row['contract_length'],
            is_bills_included=row['is_bills_included'] == 'true',
            latitude=float(row['latitude']),
            longitude=float(row['longitude']),
            amenities=None  # Not directly in CSV
        )
        apartments.append(apartment)
    
    return apartments

def load_enquiries(csv_path, profiles):
    """Load enquiries and update profile likes."""
    # Create a mapping of user_id to profile object for faster lookup
    user_to_profile = {profile.user_id: profile for profile in profiles}
    
    df = pd.read_csv(csv_path)
    
    for _, row in df.iterrows():
        user_id = row['user_id']
        property_id = row['property_id']
        
        # Add property_id to the user's likes
        if user_id in user_to_profile:
            user_to_profile[user_id].likes.append(property_id)
    
    return profiles


def build_interaction_matrix(profiles, apartments):
        """Build a user-item interaction matrix where each cell (i,j) is 1 if user i likes apartment j."""
        n_users = len(profiles)
        n_apartments = len(apartments)
        
        # Initialize with zeros
        matrix = np.zeros((n_users, n_apartments))
        
        # Fill in the likes
        for user_idx, profile in enumerate(profiles):
            for property_id in profile.apt_likes:
                if property_id in apartments.id:
                    property_idx = apartments.id[property_id]
                    matrix[user_idx, property_idx] = 1
        
        return matrix
    
def compute_user_similarity(self):
    """Compute cosine similarity between users based on their apartment preferences."""
    # If a user has no likes, their similarity will be NaN, so we handle this
    interaction_matrix_safe = self.interaction_matrix.copy()
    
    # Replace rows of zeros with a small value to avoid division by zero
    for i in range(interaction_matrix_safe.shape[0]):
        if np.sum(interaction_matrix_safe[i]) == 0:
            interaction_matrix_safe[i] = 0.0001
    
    # Calculate cosine similarity
    similarity = cosine_similarity(interaction_matrix_safe)
    
    # Set self-similarity to 0 to avoid recommending the user to themselves
    np.fill_diagonal(similarity, 0)
    
    return similarity
    
def apply_hard_filters(self, user_id):
    """Apply hard filters to find compatible matches."""
    if user_id not in self.user_id_to_index or user_id not in self.user_filters:
        return []
    
    user_filter = self.user_filters[user_id]
    user_profile = None
    
    # Find the profile for this user_id
    for profile in self.profiles:
        if profile.user_id == user_id:
            user_profile = profile
            break
    
    if not user_profile:
        return []
    
    # Get compatible users
    compatible_users = []
    
    for other_profile in self.profiles:
        # Skip self
        if other_profile.user_id == user_id:
            continue
            
        # Skip if other user doesn't have a filter
        if other_profile.user_id not in self.user_filters:
            continue
        
        other_filter = self.user_filters[other_profile.user_id]
        
        # Check budget compatibility (overlap in ranges)
        if (user_filter.apt_price_range and other_filter.apt_price_range and
            max(user_filter.apt_price_range[0], other_filter.apt_price_range[0]) >= 
            min(user_filter.apt_price_range[1], other_filter.apt_price_range[1])):
            continue
        
        # Check bedroom count compatibility
        if (user_filter.bedroom_count_range and other_filter.bedroom_count_range and
            max(user_filter.bedroom_count_range[0], other_filter.bedroom_count_range[0]) >= 
            min(user_filter.bedroom_count_range[1], other_filter.bedroom_count_range[1])):
            continue
        
        # Check property type compatibility (if both have specific preferences)
        if (user_filter.property_type != "ANY" and 
            other_filter.property_type != "ANY" and 
            user_filter.property_type != other_filter.property_type):
            continue
        
        # Check gender preferences
        if user_profile.preferred_gender not in ["ANY", None] and user_profile.preferred_gender != other_profile.gender:
            continue
            
        if other_profile.preferred_gender not in ["ANY", None] and other_profile.preferred_gender != user_profile.gender:
            continue
        
        # Check age compatibility
        if user_profile.age_range and other_profile.age_range:
            user_age = 2025 - user_profile.birth_date.year if user_profile.birth_date else 25  # Default age if missing
            other_age = 2025 - other_profile.birth_date.year if other_profile.birth_date else 25
            
            if not (user_profile.age_range[0] <= other_age < user_profile.age_range[1]):
                continue
                
            if not (other_profile.age_range[0] <= user_age < other_profile.age_range[1]):
                continue
        
        # If all checks pass, add to compatible users
        compatible_users.append(other_profile.user_id)
    
    return compatible_users

def recommend_roommates(self, user_id, top_n=5):
    """Recommend potential roommates based on collaborative filtering with hard filters."""
    if user_id not in self.user_id_to_index:
        return []
    
    # Apply hard filters first
    filtered_users = self.apply_hard_filters(user_id)
    
    if not filtered_users:
        return []
    
    # Get the user index
    user_idx = self.user_id_to_index[user_id]
    
    # Get similarity scores for filtered users
    filtered_scores = []
    for other_id in filtered_users:
        if other_id in self.user_id_to_index:
            other_idx = self.user_id_to_index[other_id]
            similarity = self.user_similarity[user_idx, other_idx]
            filtered_scores.append((other_id, similarity))
    
    # Sort by similarity score
    filtered_scores.sort(key=lambda x: x[1], reverse=True)
    
    # Return top N recommendations
    return filtered_scores[:top_n]