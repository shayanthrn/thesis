from django.db import models

class TotalNumberOfConverts(models.Model):
    id = models.CharField(default="123",max_length=5, primary_key=True)
    count = models.IntegerField(default=0)

class Content(models.Model):
    id = models.CharField(default="123",max_length=5, primary_key=True)
    content = models.TextField()

class SourceVoices(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=6,choices=[('male','male'),('female','female')])
    file = models.CharField(max_length=100)

class ConvertedVoices(models.Model):
    age = models.IntegerField()
    gender = models.CharField(max_length=6,choices=[('male','male'),('female','female')])
    to = models.CharField(max_length=4)
    source = models.CharField(max_length=100)
    file = models.CharField(max_length=100)

class Rate(models.Model):
    rate = models.IntegerField()
    phonenumber = models.CharField(max_length=20)