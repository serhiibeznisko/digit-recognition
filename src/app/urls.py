from django.urls import path
from django.contrib import admin

from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.PredictView.as_view()),
    path('ml/predict', views.predict),
    path('ml/train', views.train),
]
