import pandas as pd
from typing import List, Tuple
from helper_functions import initialize_profile_list, calculate_age, assign_profiles_to_profile_list
from generate_profiles import Profile
from pinecone import Pinecone, ServerlessSpec
import json
from manage import starting_profile  # Import Alex's profile

def profile_to_text(profile: Profile) -> str:
    """Convert profile attributes to a text string for embedding"""
    return f"""
    Name: {profile.first_name} {profile.last_name}
    Age: {calculate_age(profile.birth_date)}
    Gender: {profile.gender}
    Description: {profile.description}
    Languages: {', '.join(profile.languages)}
    Origin Country: {profile.origin_country}
    Occupation: {profile.occupation}
    Work Industry: {profile.work_industry}
    Course: {profile.course}
    Activity Hours: {profile.activity_hours}
    Smoking: {profile.smoking}
    Location: {profile.rent_location_preference}
    Budget Range: £{profile.rent_budget[0]}-£{profile.rent_budget[1]}
    Available From: {profile.available_at}
    Interests: {', '.join(profile.interests)}
    """

def upload_profiles_to_pinecone():
    print("Starting upload process...")  # Debug print
    
    # Initialize Pinecone
    PINECONE_API_KEY = "pcsk_65UBva_G8bL9ADx5Ea72DM8jTcirezWayGDHkHAi4eY3YBHXbN43tHpnfdUhUgzEbTwAKr"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    print("Pinecone initialized")  # Debug print
    
    # Create or get index
    index_name = "roommate-profiles8"
    if index_name not in pc.list_indexes():
        print(f"Creating new index: {index_name}")  # Debug print
        pc.create_index(
            name=index_name,
            dimension=1024,  # Dimension size depends on your embedding model
            metric="euclidean",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    index = pc.Index(index_name)
    print("Index accessed successfully")  # Debug print

    # Initialize profiles
    profile_objects = initialize_profile_list()
    profile_list: List[Profile] = assign_profiles_to_profile_list(starting_profile, profile_objects)
    print(f"Generated {len(profile_list)} profiles")  # Debug print
    
    # Convert profiles to embeddings and store in Pinecone
    for i, profile in enumerate(profile_list):
        print(f"Processing profile {i+1}/{len(profile_list)}")  # Debug print
        # Convert profile to text
        profile_text = profile_to_text(profile)
        
        # Generate embedding using Pinecone's embedding service
        embedding: List[float] = pc.inference.embed(
            model="multilingual-e5-large",
            inputs=[profile_text],
            parameters={"input_type": "passage", "truncate": "END"}
        )
        
        # Create metadata
        metadata = {
            "user_id": profile.user_id,
            "name": f"{profile.first_name} {profile.last_name}",
            "age": calculate_age(profile.birth_date),
            "gender": profile.gender,
            "occupation": profile.occupation,
            "location": profile.rent_location_preference,
            "budget_min": profile.rent_budget[0],
            "budget_max": profile.rent_budget[1],
            "available_at": profile.available_at
        }
        
        # Upsert to Pinecone
        index.upsert(
            vectors=[{
                "id": str(profile.user_id),
                "values": embedding[0].values,
                "metadata": metadata
            }],
            namespace="profiles"
        )

    print("Profiles have been successfully uploaded to Pinecone")
    return pc, index

def query_similar_profiles(pc: Pinecone, index, query_profile: Profile, top_k: int = 5):
    """Query Pinecone for similar profiles"""
    query_text = profile_to_text(query_profile)
    query_embedding = pc.inference.embed(
        model="multilingual-e5-large",
        inputs=[query_text],
        parameters={"input_type": "query"}
    )

    results = index.query(
        namespace="profiles",
        vector=query_embedding[0].values,
        top_k=top_k,
        include_metadata=True
    )

    print(f"\nTop {top_k} Similar Profiles:")
    for match in results.matches:
        metadata = match.metadata
        print(f"\nProfile ID: {match.id}")
        print(f"Similarity Score: {match.score:.3f}")
        print(f"Name: {metadata['name']}")
        print(f"Age: {metadata['age']}")
        print(f"Gender: {metadata['gender']}")
        print(f"Occupation: {metadata['occupation']}")
        print(f"Location: {metadata['location']}")
        print(f"Budget Range: £{metadata['budget_min']}-£{metadata['budget_max']}")
        print(f"Available From: {metadata['available_at']}")
        print("-" * 50) 

if __name__ == "__main__":
    print("Starting main execution...")  # Debug print
    try:
        pc, index = upload_profiles_to_pinecone()
        print("Profiles uploaded successfully")  # Debug print
        
        print("\nComparing against this profile:")
        print(profile_to_text(starting_profile))  
        print("Test profile initialized")  # Debug print
        
        query_similar_profiles(pc, index, starting_profile) 
        print("Query completed")  # Debug print
    except Exception as e:
        print(f"An error occurred: {str(e)}")  # Error handling