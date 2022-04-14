from django.shortcuts import render
from django.views import View

class MainView(View):

    def get(self, request):
        context = {}
        return render(request, 'VoiceConversion/index.html',context)