from django.urls import path

from . import views

urlpatterns = [
    path('train/result/download_model', views.download_model),
    path('train/result/download_vector', views.download_vector),
    path('train/result', views.train_result, name='train_result'),
    path('train', views.train, name='train'),
    path('result', views.result, name='result'),
    path('', views.index, name='index'),
]