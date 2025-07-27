# recommender/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.recommend_view, name='recommend'),
]
