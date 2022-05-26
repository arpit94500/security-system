from django.contrib import admin
from django.urls import path
from home import views
from django.contrib import admin
from django.urls import path,include

urlpatterns = [
    path('', views.index, name='index'),
    # path('Login/', views.Login, name='Login'),
    # path('Register/', views.Register, name='Register'),
    path('index/',views.index, name='index'),
    path('logout/',views.logout, name='logout'),
    path('click',views.click, name='click'),
    path('sections/add/',views.add, name='add'),
    path('sections/category/',views.category, name='category'),
    path('sections/view/',views.view, name='view'),
    path('sections/checklog/',views.checklog, name='checklog'),
    path('sections/<int:num>', views.section, name='section'),
    path('video_feed1', views.video_feed1, name='video_feed1'),
]
