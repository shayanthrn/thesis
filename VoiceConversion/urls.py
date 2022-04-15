from django.urls import path

from .views import *

app_name = 'VoiceConversion'

urlpatterns = [
    path('', MainView.as_view(), name='main'),
    path('convertWav',convertWavView.as_view(),name='convert wav')
]