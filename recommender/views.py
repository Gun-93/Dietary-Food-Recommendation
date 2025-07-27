from django.shortcuts import render
from .recommend import recommend_recipes
import os
import joblib

# BASE_DIR points to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Point to food_recommendation_system/models
MODEL_DIR = os.path.join(BASE_DIR, 'models')
MODEL_PATH = os.path.join(MODEL_DIR, 'knn_model.pkl')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')

# Load models
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def recommend_view(request):
    if request.method == 'POST':
        ingredients = request.POST.get('ingredients', '').strip()
        user_type = request.POST.get('user_type', '').strip()
        food_type = request.POST.get('food_type', '').strip()
        allergics = request.POST.get('allergics', '').strip()
        gender = request.POST.get('gender', '').strip()

        if not ingredients:
            return render(request, 'form.html', {
                'error': "Please enter ingredients to get recommendations."
            })
        try:
            # Pass the loaded model and vectorizer into your function
            results = recommend_recipes(
                ingredients=ingredients,
                user_type=user_type,
                c_type=food_type,
                allergics=allergics,
                gender=gender,
                model=model,
                vectorizer=vectorizer
            )
            recipes = results.to_dict(orient='records') if not results.empty else []
            return render(request, 'results.html', {
                'recipes': recipes
            })
        except Exception as e:
            return render(request, 'form.html', {
                'error': f"Error while recommending: {str(e)}"
            })

    return render(request, 'form.html')
