# benchmarks/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('accelerators/', views.accelerators, name='accelerators'),
]
