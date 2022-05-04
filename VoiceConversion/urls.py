from django.urls import path

from .views import *

app_name = 'VoiceConversion'

urlpatterns = [
    path('', MainView.as_view(), name='main'),
    path('eval/', EvalView.as_view(), name='eval'),
    path('about/', AboutView.as_view(), name='about'),
    path('contactus/', ContactView.as_view(), name='contact'),
]