from django.contrib import admin
from home.models import Face, Entry, register, Save, accuracy, cat, LastFace, super

# Register your models here.
admin.site.register(Face)
admin.site.register(Entry)
admin.site.register(register)
admin.site.register(Save)
admin.site.register(cat)
admin.site.register(accuracy)
admin.site.register(LastFace)
admin.site.register(super)
