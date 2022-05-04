from django.shortcuts import redirect, render
from django.views import View
from .inference import Inference
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from .models import *
from django.utils import timezone
from pydub import AudioSegment



Inferencer = Inference()
# done
class MainView(View):

    def __init__(self):
        
        super().__init__()

    def get(self, request):
        countobj, created = TotalNumberOfConverts.objects.get_or_create(id="123")
        return render(request, 'VoiceConversion/index.html',context={'totalusers':countobj.count,'speaker':range(1,65),'age':range(4,80),'select':"1"})

    def post(self, request):
        filesize =(request.FILES['audio'].size/1024)/1024 # in MB
        if(filesize>2):
            return HttpResponseForbidden("file size not allowed")
        else:
            if(request.FILES['audio'].content_type=="audio/wav"):
                timestamp = int(timezone.now().timestamp())
                filename = f'./sources/{timestamp}.wav'
                targetfile = f'./static/converted/{timestamp}.wav'
                file = open(filename, 'wb')
                file.write(request.FILES['audio'].file.read())
            else:
                # TODO convert format
                return HttpResponseForbidden("unsupported format file")

            SourceVoices.objects.create(age=request.POST.get('age','20'),gender=request.POST.get('gender','male'),file = filename)
            try:
                Inferencer.inference(filename,request.POST['speaker'],targetfile)
                ConvertedVoices.objects.create(file=targetfile,to=request.POST['speaker'],age=request.POST.get('age','20'),gender=request.POST.get('gender','male'),source =filename)
            except:
                return HttpResponseBadRequest("Out of Memory. Please try again later")
            countobj = TotalNumberOfConverts.objects.filter(id="123").first()
            count = countobj.count + 1
            countobj.count = count
            countobj.save()
            return render(request, 'VoiceConversion/index.html',context={'totalusers':count,'speaker':range(1,65),'age':range(4,80),'select':"1","target":targetfile[1:]})
        

class EvalView(View):

    def get(self, request):
        return render(request, 'VoiceConversion/eval.html',context={'select':"2"})
    
    def post(self, request):
        phonenumber=request.POST.get("phonenumber","unkown")
        rate = int(request.POST.get("rate","-1"))
        if(rate<1 or rate>5):
            return HttpResponseBadRequest("Malicious user")
        else:
            Rate.objects.update_or_create(phonenumber=phonenumber,defaults={"rate":rate})
            return render(request, 'VoiceConversion/eval_success.html',context={'select':"2"})
        
# Done
class AboutView(View):

    def get(self, request):
        contentobj, created = Content.objects.get_or_create(id="123")
        return render(request, 'VoiceConversion/about.html',context={'content':contentobj.content,'select':"3"})

# Done
class ContactView(View):

    def get(self, request):
        return render(request, 'VoiceConversion/contact.html',context={'select':"4"})