import os
import pandas as pd
import numpy as np
from datetime import datetime
from generate_profiles2 import Profile
from manage2 import XGBoostRecommender
import random
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb

def load_profiles_from_csv(csv_path):
    """Load profiles from CSV file"""
    df = pd.read_csv(csv_path)
    profiles = {}
    
    # Print available columns for debugging
    print("\nAvailable columns in CSV:")
    print(df.columns.tolist())
    
    for _, row in df.iterrows():
        try:
            profile = Profile(
                user_id=row['user_id'],
                first_name=row.get('first_name', ''),
                last_name=row.get('last_name', ''),
                birth_date=pd.to_datetime(row['birth_date']),
                is_verified=row.get('is_verified', False),
                gender=row.get('gender', ''),
                description=None,
                languages=row.get('languages', '').split(',') if pd.notna(row.get('languages')) else [],
                origin_country=row.get('origin_country', ''),
                occupation=row.get('occupation', ''),
                work_industry=row.get('work_industry') if pd.notna(row.get('work_industry')) else None,
                university_id=row.get('university_id') if pd.notna(row.get('university_id')) else None,
                course=row.get('course_id') if pd.notna(row.get('course_id')) else None,  # Changed from 'course' to 'course_id'
                sexual_orientation=row.get('sexual_orientation', ''),
                pets=row.get('pets') if pd.notna(row.get('pets')) else None,
                activity_hours=row.get('activity_hours', ''),
                smoking=row.get('smoking', ''),
                extrovert_level=row.get('extrovert_level', 0),
                cleanliness_level=row.get('cleanliness_level', 0),
                partying_level=row.get('partying_level', 0),
                sex_living_preference=None,
                rent_location_preference=None,
                age_preference=None,
                rent_budget=None,
                last_filter_processed_at=None,
                available_at=None,
                roommate_count_preference=None,
                interests=[]
            )
            profiles[row['user_id']] = profile
        except Exception as e:
            print(f"Error creating profile for row: {row}")
            print(f"Error: {str(e)}")
            continue
    
    return profiles

def generate_training_data(profiles: dict[int, Profile], likes_df: pd.DataFrame, negative_ratio: float = 1.0): #negaive_ratio is a parameter
    """
    Generate training data with positive cases from likes and negative cases from non-likes
    negative_ratio: number of negative samples per positive sample
    """
    training_data = []
    
    # Add positive samples from likes
    for _, row in likes_df.iterrows():
        viewer_id = row['profile_id_1']
        liked_id = row['profile_id_2']
        
        if viewer_id in profiles and liked_id in profiles:
            training_data.append((
                profiles[viewer_id],
                profiles[liked_id],
                1  # positive case
            ))
    
    # Generate negative samples
    all_user_ids = list(profiles.keys())
    num_negative_samples = int(len(training_data) * negative_ratio)
    
    negative_pairs = set()
    positive_pairs = set((row['profile_id_1'], row['profile_id_2']) for _, row in likes_df.iterrows())
    
    # Pre-calculate valid candidate pairs to improve performance
    attempts = 0
    max_attempts = num_negative_samples * 10  # Prevent infinite loops
    
    while len(negative_pairs) < num_negative_samples and attempts < max_attempts:
        viewer_id = random.choice(all_user_ids)
        candidate_id = random.choice(all_user_ids)
        
        if viewer_id != candidate_id and (viewer_id, candidate_id) not in positive_pairs:
            negative_pairs.add((viewer_id, candidate_id))
            training_data.append((
                profiles[viewer_id],
                profiles[candidate_id],
                0  # negative case
            ))
        attempts += 1
    
    if attempts >= max_attempts:
        print(f"Warning: Could only generate {len(negative_pairs)} negative samples out of {num_negative_samples} requested")
    
    return training_data

def print_feature_importance(model):
    """
    Print and plot the feature importance of the trained XGBoost model.
    """
    # Define feature names in the exact order as they appear in the feature vector
    feature_names = [
        'budget_overlap',
        'viewer_age',
        'candidate_age',
        'age_difference',
        'origin_country_match',
        'course_match',
        'occupation_match',
        'work_industry_match',
        'smoking_match',
        'activity_hours_match',
        'university_match',
        'extrovert_level_difference',
        'cleanliness_level_difference',
        'partying_level_difference',
        'gender_match',
        'sexual_orientation_compatibility',
        'pets_match',
        'languages_overlap'
    ]
    
    print("\n[DEBUG] Starting print_feature_importance function.")
    
    try:
        # Get feature importances
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif isinstance(model, xgb.XGBClassifier):
            booster = model.get_booster()
            importances = [booster.get_score(importance_type='gain').get(f'f{i}', 0) for i in range(len(feature_names))]
        else:
            print("[DEBUG] Cannot retrieve feature importances")
            return

        # Normalize the importances
        total_importance = sum(importances) if sum(importances) > 0 else 1
        normalized_scores = [score/total_importance for score in importances]

        # Print the feature importances
        print("\nFeature Importance Scores:")
        print("-" * 60)
        print(f"{'Feature Name':<40} {'Importance':>10}")
        print("-" * 60)
        
        # Create sorted pairs of (feature_name, importance)
        importance_pairs = list(zip(feature_names[:len(importances)], normalized_scores))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # Print each feature and its importance
        for name, importance in importance_pairs:
            print(f"{name:<40} {importance:>10.3f}")

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.bar([x[0] for x in importance_pairs], [x[1] for x in importance_pairs], color='skyblue')
        plt.xlabel('Features')
        plt.ylabel('Normalized Importance')
        plt.title('Feature Importance')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()
        
        # Additionally, using XGBoost's built-in plot_importance
        booster = model.get_booster()
        plt.figure(figsize=(10, 6))
        xgb.plot_importance(booster, max_num_features=10, importance_type='gain', xlabel='Gain', title='Feature Importance (Gain)')
        plt.show()
        
    except Exception as e:
        print(f"\n[ERROR] An exception occurred in print_feature_importance: {str(e)}")
        import traceback
        traceback.print_exc()

    print("\n[DEBUG] Completed print_feature_importance function.")

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the XGBoost model using various metrics
    """
    # Get predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    
    # Print metrics
    
    # Create confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'auc_roc': auc_roc
    }

def main():
    # Get the current directory where train_from_csv.py is located
    current_dir = os.path.dirname(__file__)
    
    # Construct paths to CSV files
    profiles_path = os.path.join(current_dir, 'profile.csv')
    likes_path = os.path.join(current_dir, 'profile_like.csv')
    
    # Load data
    profiles = load_profiles_from_csv(profiles_path)
    likes = pd.read_csv(likes_path)
    
    print(f"Loaded {len(profiles)} profiles and {len(likes)} likes")
    
    # Generate training data
    training_data = generate_training_data(profiles, likes)
    
    # Initialize recommender first
    recommender = XGBoostRecommender()
    
    # Then prepare features and labels
    X = [recommender.create_feature_vector(viewer, candidate) 
         for viewer, candidate, _ in training_data]
    y = [label for _, _, label in training_data]
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        np.array(X),
        np.array(y),
        test_size=0.2,
        random_state=42
    )
    
    # Train the model
    recommender.model.fit(
        X_train, 
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Comment out the evaluate_model section temporarily
    # metrics = evaluate_model(recommender.model, X_test, y_test)
    # print("\nModel Performance Metrics:")
    # print("-" * 50)
    # print(f"Accuracy:  {metrics['accuracy']:.3f}")
    # print(f"Precision: {metrics['precision']:.3f}")
    # print(f"Recall:    {metrics['recall']:.3f}")
    # print(f"AUC-ROC:   {metrics['auc_roc']:.3f}")
    
    # Save the trained model
    recommender.save_model('trained_model.json')
    
    # Print feature importances
    print("\nGenerating Feature Importance Analysis:")
    print("-" * 50)
    print_feature_importance(recommender.model)
    
    print("\n[DEBUG] End of main function.")

if __name__ == "__main__":
    main() 