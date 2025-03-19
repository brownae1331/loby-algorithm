import csv
from typing import List, Dict, Tuple, Set
import pandas as pd
from datetime import datetime, date, timedelta
from typing import List, Tuple, Dict
from generate_profiles3 import Profile, Apartment, UserPropertyFilter
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import NMF

def calculate_age(birth_date):
    """Calculate age from birth date."""
    today = date.today()
    return (
        today.year
        - birth_date.year
        - ((today.month, today.day) < (birth_date.month, birth_date.day))
    )

class CollaborativeFilteringMatcher:
    def __init__(self, profiles, apartments, user_apt_filters, similarity_method="jaccard"):
        self.profiles = profiles
        self.apartments = apartments
        self.apartment_filters = user_apt_filters  
        
        # Create mappings for faster lookup
        self.user_id_to_index = {profile.user_id: i for i, profile in enumerate(self.profiles)}
        self.apartment_id_to_index = {apartment.id: i for i, apartment in enumerate(self.apartments)}
        self.index_to_user_id = {i: profile.user_id for i, profile in enumerate(self.profiles)}
        self.index_to_apartment_id = {i: apartment.id for i, apartment in enumerate(self.apartments)}
        
        # Build the user-item interaction matrix
        self.interaction_matrix = self._build_interaction_matrix()
        
        # Compute user similarity matrix using the specified method
        self.similarity_method = similarity_method
        if similarity_method == "matrix_factorization":
            self.user_similarity = self._compute_matrix_factorization()
        else:
            self.user_similarity = self.compute_similarity(method=similarity_method)

    def _build_interaction_matrix(self):
        """Build a user-item interaction matrix where each cell (i,j) is 1 if user i likes apartment j."""
        n_users = len(self.profiles)
        n_apartments = len(self.apartments)
        
        # Initialize with zeros
        matrix = np.zeros((n_users, n_apartments))
        
        # Fill in the likes
        for user_idx, profile in enumerate(self.profiles):
            for property_id in profile.apt_likes:
                if property_id in self.apartment_id_to_index:
                    property_idx = self.apartment_id_to_index[property_id]
                    matrix[user_idx, property_idx] = 1
        
        return matrix

    def _compute_jaccard_similarity(self):
        """
        Compute Jaccard similarity between users based on apartment likes.
        Jaccard similarity = (size of intersection) / (size of union)
        This method focuses only on liked apartments, ignoring the absence of likes.
        """
        n_users = len(self.profiles)
        similarity = np.zeros((n_users, n_users))
        
        # For each pair of users
        for i in range(n_users):
            # Get the set of apartments liked by user i
            user_i_likes = set(self.profiles[i].apt_likes)
            
            for j in range(n_users):
                if i == j:
                    continue  # Skip self-comparison
                    
                # Get the set of apartments liked by user j
                user_j_likes = set(self.profiles[j].apt_likes)
                
                # Calculate Jaccard similarity: intersection size / union size
                intersection_size = len(user_i_likes.intersection(user_j_likes))
                union_size = len(user_i_likes.union(user_j_likes))
                
                # Avoid division by zero
                if union_size > 0:
                    similarity[i, j] = intersection_size / union_size
        
        return similarity
    
    def _compute_matrix_factorization(self, n_components=10):
        """
        Compute user similarity using matrix factorization.
        
        Parameters:
        - n_components: Number of latent factors to use
        
        Returns:
        - A similarity matrix
        """
        # Initialize and fit the NMF model with more iterations and better initialization
        model = NMF(n_components=n_components, 
                   init='nndsvd',  # Use SVD-based initialization
                   max_iter=200,   # More iterations
                   random_state=0)
        
        # Add small noise to avoid zero rows
        interaction_matrix_noisy = self.interaction_matrix + np.random.normal(0, 0.01, self.interaction_matrix.shape)
        interaction_matrix_noisy = np.maximum(interaction_matrix_noisy, 0)  # Ensure non-negative
        
        # Fit the model
        W = model.fit_transform(interaction_matrix_noisy)
        H = model.components_
        
        # Compute user similarity using the reconstructed matrix
        similarity = np.dot(W, W.T)
        
        # Normalize the similarity scores
        similarity = similarity / np.max(similarity)
        
        # Set self-similarity to 0
        np.fill_diagonal(similarity, 0)
        
        # Add a small baseline similarity for users with no interactions
        baseline_similarity = 0.1
        similarity = np.maximum(similarity, baseline_similarity)
        
        return similarity

    def compute_similarity(self, method="jaccard"):
        """
        Compute user similarity using the specified method.
        
        Parameters:
        - method: The similarity method to use ('cosine', 'jaccard', or 'matrix_factorization')
        
        Returns:
        - A similarity matrix
        """
        if method.lower() == "jaccard":
            return self._compute_jaccard_similarity()
        elif method.lower() == "matrix_factorization":
            return self._compute_matrix_factorization()
        else:  # Default to cosine similarity
            return self._compute_user_similarity()
    
    def print_interaction_matrix(self):
        """Print the user-item interaction matrix in a readable format."""
        print("\nInteraction Matrix (Users x Apartments):")
        print("--------------------------------------")
        print(f"Matrix shape: {self.interaction_matrix.shape} (Rows: Users, Columns: Apartments)")
        
        # Print column headers (apartment IDs)
        max_display_cols = min(10, self.interaction_matrix.shape[1])
        apt_headers = [self.index_to_apartment_id[i] for i in range(max_display_cols)]
        print(f"{'User/Apt':>10}", end="")
        for apt_id in apt_headers:
            print(f"{apt_id:>6}", end="")
        print()
        
        # Print matrix rows
        max_display_rows = min(20, self.interaction_matrix.shape[0])
        for i in range(max_display_rows):
            user_id = self.index_to_user_id[i]
            print(f"{user_id:>10}", end="")
            for j in range(max_display_cols):
                print(f"{int(self.interaction_matrix[i, j]):>6}", end="")
            print()
        
        # Print matrix statistics
        total_interactions = np.sum(self.interaction_matrix)
        density = total_interactions / (self.interaction_matrix.shape[0] * self.interaction_matrix.shape[1])
        print("\nMatrix Statistics:")
        print(f"Total interactions: {int(total_interactions)}")
        print(f"Matrix density: {density:.4f} (proportion of non-zero entries)")
        
        # Show distribution of interactions per user
        interactions_per_user = np.sum(self.interaction_matrix, axis=1)
        avg_interactions = np.mean(interactions_per_user)
        max_interactions = np.max(interactions_per_user)
        print(f"Average interactions per user: {avg_interactions:.2f}")
        print(f"Maximum interactions per user: {int(max_interactions)}")
    
    def print_similarity_matrix(self):
        """Print the user similarity matrix in a readable format."""
        print(f"\nUser Similarity Matrix (using {self.similarity_method.upper()} similarity):")
        print("----------------------")
        print(f"Matrix shape: {self.user_similarity.shape}")
        
        # Print a subset of the matrix
        max_display = min(10, self.user_similarity.shape[0])
        
        # Print column headers (user IDs)
        print(f"{'User/User':>10}", end="")
        for i in range(max_display):
            user_id = self.index_to_user_id[i]
            print(f"{user_id:>6}", end="")
        print()
        
        # Print matrix rows
        for i in range(max_display):
            user_id = self.index_to_user_id[i]
            print(f"{user_id:>10}", end="")
            for j in range(max_display):
                print(f"{self.user_similarity[i, j]:>6.2f}", end="")
            print()
        
        # Print matrix statistics
        print("\nSimilarity Matrix Statistics:")
        non_zeros = np.count_nonzero(self.user_similarity)
        total_cells = self.user_similarity.shape[0] * self.user_similarity.shape[1]
        print(f"Non-zero similarities: {non_zeros} out of {total_cells} ({non_zeros/total_cells:.2%})")
        
        # Remove diagonal (which we set to zero)
        similarity_no_diag = self.user_similarity.copy()
        np.fill_diagonal(similarity_no_diag, np.nan)
        
        avg_sim = np.nanmean(similarity_no_diag)
        max_sim = np.nanmax(similarity_no_diag)
        print(f"Average similarity: {avg_sim:.4f}")
        print(f"Maximum similarity: {max_sim:.4f}")
    
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
    
    def apply_hard_filters(self, user_id):
        """Apply hard filters to find compatible matches based on user preferences."""
        if user_id not in self.user_id_to_index:
            return []
        
        # Find the profile for this user_id
        viewer_profile = None
        for profile in self.profiles:
            if profile.user_id == user_id:
                viewer_profile = profile
                break
        
        if not viewer_profile:
            return []
        
        filtered_profiles = []
        
        # Track filter failures for reporting
        failed_date = failed_budget = failed_gender = failed_age = 0
        
        # Parse viewer's available date
        viewer_available_at = parse_date(viewer_profile.available_at)
        
        for profile in self.profiles:
            # Skip if it's the same user
            if profile.user_id == viewer_profile.user_id:
                continue
            
            # 1. Date range check (within 2 weeks)
            if viewer_available_at and profile.available_at:
                profile_available_at = parse_date(profile.available_at)
                if profile_available_at:
                    # Convert to datetime.date objects for comparison
                    if isinstance(viewer_available_at, datetime):
                        viewer_available_at = viewer_available_at.date()
                    if isinstance(profile_available_at, datetime):
                        profile_available_at = profile_available_at.date()
                        
                    if not (
                        (viewer_available_at - timedelta(days=14))
                        <= profile_available_at
                        <= (viewer_available_at + timedelta(days=14))
                    ):
                        failed_date += 1
                        continue
            
            # 2. Budget overlap check
            if viewer_profile.rent_budget_range and profile.rent_budget_range:
                viewer_min, viewer_max = viewer_profile.rent_budget_range
                candidate_min, candidate_max = profile.rent_budget_range
                
                # Check if there's no overlap in budget ranges
                if viewer_max < candidate_min or candidate_max < viewer_min:
                    failed_budget += 1
                    continue
            
            # 3. Age range check
            if hasattr(viewer_profile, 'age_range') and viewer_profile.age_range and profile.birth_date:
                min_age, max_age = viewer_profile.age_range
                candidate_age = calculate_age(profile.birth_date)
                
                if candidate_age < min_age or candidate_age > max_age:
                    failed_age += 1
                    continue
                    
            # Check if candidate's age preferences include viewer
            if hasattr(profile, 'age_range') and profile.age_range and viewer_profile.birth_date:
                min_age, max_age = profile.age_range
                viewer_age = calculate_age(viewer_profile.birth_date)
                
                if viewer_age < min_age or viewer_age > max_age:
                    failed_age += 1
                    continue
            
            # 4. Gender preference check
            if viewer_profile.preferred_gender and profile.gender:
                # Check if viewer's preference matches candidate's gender
                viewer_accepts_candidate = (
                    viewer_profile.preferred_gender == "ANY" or 
                    viewer_profile.preferred_gender == profile.gender
                )
                
                if not viewer_accepts_candidate:
                    failed_gender += 1
                    continue
                    
            # Check if candidate's gender preference matches viewer
            if profile.preferred_gender and viewer_profile.gender:
                # Check if candidate's preference matches viewer's gender
                candidate_accepts_viewer = (
                    profile.preferred_gender == "ANY" or 
                    profile.preferred_gender == viewer_profile.gender
                )
                
                if not candidate_accepts_viewer:
                    failed_gender += 1
                    continue
            
            # If passed all hard filters, add to filtered profiles
            filtered_profiles.append(profile.user_id)
        
        # Print filter stats
        # print(f"\nFilter results for user {viewer_profile.user_id}:")
        # print(f"Failed date range: {failed_date}")
        # print(f"Failed budget overlap: {failed_budget}")
        # print(f"Failed gender preference: {failed_gender}")
        # print(f"Failed age preference: {failed_age}")
        # print(f"Passed all filters: {len(filtered_profiles)}")
        
        return filtered_profiles

    def apply_apartment_filters(self, user_id):
        """Apply apartment filters to find compatible properties based on user preferences."""
        if user_id not in self.user_id_to_index or user_id not in self.apartment_filters:
            return []
        
        apt_filter = self.apartment_filters[user_id]
        compatible_apartments = []
        
        for apartment in self.apartments:
            # Check price range
            if apt_filter.apt_price_range:
                if not (apt_filter.apt_price_range[0] <= apartment.cost < apt_filter.apt_price_range[1]):
                    continue
            
            # Check bedroom count
            if apt_filter.bedroom_count_range:
                if not (apt_filter.bedroom_count_range[0] <= apartment.bedroom_count < apt_filter.bedroom_count_range[1]):
                    continue
            
            # Check property type
            if apt_filter.property_type != "ANY" and apt_filter.property_type != apartment.property_type:
                continue
            
            # If all checks pass, add to compatible apartments
            compatible_apartments.append(apartment.id)
        
        return compatible_apartments
    



####### cosine similarity (not being used) #######
    def _compute_user_similarity(self):
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
########################################################

def load_profiles(csv_path):
    """Load profiles from CSV file."""
    profiles = []
    user_apt_filters = {}
    
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
            university_id=int(row['university_id']) if not pd.isna(row['university_id']) and row['university_id'] != '' else None,
            course_id=int(row['course_id']) if not pd.isna(row['course_id']) and row['course_id'] != '' else None,
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
            likes=[],  
            apt_likes=[] # Will fill this later from enquiries
        )
        profiles.append(profile)
        
        # Create UserPropertyFilter object
        filter_created_at = datetime.strptime(row['property_filter_created_at'].split('+')[0].strip(), '%Y-%m-%d %H:%M:%S.%f')
        filter_obj = UserPropertyFilter(  # Create a filter object
            user_id=row['user_id'],
            apt_price_range=parse_range(row['property_rent_budget']),
            bedroom_count_range=parse_range(row['bedroom_count_range']),
            property_type=row['property_type'],
            created_at=filter_created_at
        )
        user_apt_filters[row['user_id']] = filter_obj  # Add to dictionary with user_id as key
    
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
            user_to_profile[user_id].apt_likes.append(property_id)
    
    return profiles



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
