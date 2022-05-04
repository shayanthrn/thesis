from django.contrib import admin
from .models import *
# Register your models here.
admin.site.register(TotalNumberOfConverts)
admin.site.register(Content)
admin.site.register(SourceVoices)
admin.site.register(ConvertedVoices)
admin.site.register(Rate)