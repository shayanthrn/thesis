from django.shortcuts import redirect, render
from django.views import View
from .inference import Inference
from django.http import HttpResponse, HttpResponseBadRequest, HttpResponseForbidden
from .models import *
from django.utils import timezone
from pydub import AudioSegment
from django.utils.decorators import method_decorator
from ratelimit.decorators import ratelimit



Inferencer = Inference()


class AssessmentView(View):

    def get(self,request):
        rates=Rate.objects.all()
        sum = 0
        for rate in rates:
            sum+= rate.rate
        return HttpResponse(sum/len(rates))
        
    def post(self,request):
        rates=Rate.objects.all()
        sum = 0
        for rate in rates:
            sum+= rate.rate
        return HttpResponse(sum/len(rates))

class MainView(View):

    def __init__(self):
        super().__init__()

    
    def get(self, request):
        countobj, created = TotalNumberOfConverts.objects.get_or_create(id="123")
        return render(request, 'VoiceConversion/index.html',context={'totalusers':countobj.count,'speaker':range(1,65),'age':range(4,80),'select':"1"})

    @method_decorator(ratelimit(key='ip', rate='10/m', method='POST',block=True))
    def post(self, request):
        filesize =(request.FILES['audio'].size/1024)/1024 # in MB
        if(filesize>2):
            return HttpResponseForbidden("file size not allowed")
        else:
            timestamp = int(timezone.now().timestamp())
            filename = f'./sources/{timestamp}.wav'
            targetfile = f'./static/converted/{timestamp}.wav'
            if(request.FILES['audio'].content_type=="audio/wav"):
                file = open(filename, 'wb')
                file.write(request.FILES['audio'].file.read())
            elif(request.FILES['audio'].content_type=="audio/mpeg" or request.FILES['audio'].content_type=="audio/mpeg3"):
                file = open(f"./temp/{timestamp}.mp3", 'wb')
                file.write(request.FILES['audio'].file.read())
                sound = AudioSegment.from_mp3(f"./temp/{timestamp}.mp3")
                sound.export(filename, format="wav")
            else:
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