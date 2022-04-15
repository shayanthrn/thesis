from django.shortcuts import redirect, render
from django.views import View
from .inference import inference
from django.http import FileResponse

class MainView(View):

    def get(self, request):
        return render(request, 'VoiceConversion/index.html',context={'data':range(1,66)})


class convertWavView(View):

    def post(self, request):
        file = open('./source.wav', 'wb')
        file.write(request.FILES['source'].file.read())
        inference('./source.wav',request.POST['speaker'])
        response = FileResponse(open("./converted.wav", 'rb'))
        return response

    def get(self, request):
        return redirect("/")