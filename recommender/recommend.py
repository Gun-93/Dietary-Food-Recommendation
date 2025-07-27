# Required libraries for data handling and text-based similarity search
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import NearestNeighbors
import joblib
import os

# Load original dataset
df = pd.read_csv('recommender/recipes1.csv')
df['Describe'] = df['Describe'].fillna("")

# Pre-fit the vectorizer and model on the full dataset
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['Describe'])

# Train the Nearest Neighbors model
model = NearestNeighbors(n_neighbors=10, metric='cosine')
model.fit(X)

# ✅ Ensure 'models' directory exists before saving
os.makedirs('models', exist_ok=True)

# Save the trained model and vectorizer for deployment
joblib.dump(model, 'models/knn_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

# Define the recommendation function
def recommend_recipes(ingredients, user_type=None, c_type=None, allergics=None, 
        gender=None, top_n=5, 
         model=None, vectorizer=None):

    # Step 1: Vectorize user input
    user_vec = vectorizer.transform([ingredients])
    # Step 2: Run KNN on full dataset
    distances, indices = model.kneighbors(user_vec)
    results = []

    for idx, dist in zip(indices[0], distances[0]):
        if dist >= 0.6:
            continue  # Skip irrelevant
        recipe = df.iloc[idx]

        # Filter by diet
        if user_type:
            user_type = user_type.lower()
            if user_type == "vegetarian" and recipe["Veg_Non"].lower() != "veg":
                continue
            elif user_type == "non-vegetarian" and recipe["Veg_Non"].lower() != "non-veg":
                continue

        # Filter by category
        if c_type and c_type.lower() != recipe["C_Type"].lower():
            continue

        # Filter by allergens
        if allergics:
            allergy_list = [x.strip().lower() for x in allergics.split(',')]
            if any(allergen in recipe["Describe"].lower() for allergen in allergy_list):
                continue

        results.append(recipe)

        if len(results) >= top_n:
            break

    # Step 3: Final return
    if results:
        output = pd.DataFrame(results)[['Name', 'Describe']]
        if gender:
            output['Note'] = f"Suggested for gender: {gender}"
        return output
    else:
        return pd.DataFrame([{
            'Name': 'No relevant match found',
            'Describe': 'Try new ingredients or filters. No close match found.',
            'Note': f"Requested filters: user_type={user_type}, category={c_type}, allergics={allergics}, gender={gender}"
        }])
