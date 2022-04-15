from django.shortcuts import render
from django.views import View

class MainView(View):

    def get(self, request):
        return render(request, 'VoiceConversion/index.html',context={'data':range(1,66)})


class convertWavView(View):

    def post(self, request):
        pass