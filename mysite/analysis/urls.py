from django.urls import path

from . import views

urlpatterns = [
    path('result', views.result, name='result'),
    path('', views.index, name='index'),
]