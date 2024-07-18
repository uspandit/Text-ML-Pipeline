from django.urls import path
from . import views

urlpatterns = [
    path('result/', views.results, name='result'),
    path('download/<path:model_path>/', views.download_model, name='download_model'),
    path('', views.index, name='index'),
]
