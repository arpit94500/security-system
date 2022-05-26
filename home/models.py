from django.db import models
from datetime import datetime
import os

class LastFace(models.Model):
	last_face = models.CharField(max_length=20, null=True)
	def __str__(self):
		return self.last_face

class Face(models.Model):
	name = models.CharField(max_length=50)
	rank = models.CharField(max_length=50)
	number = models.CharField(max_length=12)
	snumber = models.CharField(max_length=12)
	adharno = models.CharField(max_length=50)
	cat = models.CharField(max_length=50)
	blacklist = models.CharField(max_length=50)
	gender = models.CharField(max_length=50)
	username=models.TextField(null=True)
	date = models.DateField(null=True)
	time = models.TimeField(null=True)

	def __str__(self):
		return self.name

	def num_photos(self):
		try:
			DIR = f"static/dataset/{self.name}"
			img_count = len(os.listdir(DIR))
			return img_count
		except:
			return 0

class cat(models.Model):
	category = models.CharField(max_length=50, null=True)
	date = models.DateField(null=True)
	time = models.TimeField(null=True)

	def __str__(self):
		return self.category
	
class super(models.Model):
	supervisor = models.CharField(max_length=50, null=True)
	token = models.IntegerField(null=True)
	date = models.DateField(null=True)
	time = models.TimeField(null=True)
	
	def __str__(self):
		return self.supervisor
	
class Entry(models.Model):
	srno = models.CharField(max_length=50)
	Name = models.CharField(max_length=50)
	place = models.CharField(max_length=12)
	purpose = models.CharField(max_length=50)
	supervisor = models.CharField(max_length=50)
	def __str__(self):
		return self.srno


class register(models.Model):
	username = models.CharField(max_length=50)
	password = models.CharField(max_length=50)
	date = models.DateField(null=True)
	time = models.TimeField(null=True)

class login(models.Model):
	username=models.CharField(max_length=50)
	password = models.CharField(max_length=50)
	datein = models.DateField(null=True)
	timein = models.TimeField(null=True)
	dateout = models.DateField(null=True)
	timeout = models.TimeField(null=True)

class Save(models.Model):
	# id = models.IntegerField(primary_key=True)
	name = models.CharField(max_length=50)
	rank = models.CharField(max_length=50)
	number = models.CharField(max_length=12)
	adharno = models.CharField(max_length=12)
	snumber = models.CharField(max_length=50)
	cat = models.CharField(max_length=50)
	blacklist = models.CharField(max_length=50)
	place = models.CharField(max_length=50, null=True)
	supervisor=models.CharField(max_length=50, null=True)
	token = models.CharField(max_length=50, null=True)
	timein=models.TimeField(null=True)
	datein = models.DateField(null=True)
	timeout=models.TimeField(null=True)
	dateout = models.DateField(null=True)
	def __str__(self):
		return self.name

class accuracy(models.Model):
	name = models.CharField(max_length=50)
	correct = models.IntegerField()
	incorrect = models.IntegerField()
	unknown = models.IntegerField()


