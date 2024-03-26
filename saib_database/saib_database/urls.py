# saib_database/urls.py

from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('benchmarks/', include('benchmarks.urls')),
]
