from django.urls import path
from .views import *

urlpatterns = [
    path('filter/<int:id>/', filter, name='filter'),
    path('', video_list, name='video_list'),
    path('upload_video/', upload_video, name='upload_video'),
    path('video_filter/<int:id>/', video_filter, name='video_filter'),
]