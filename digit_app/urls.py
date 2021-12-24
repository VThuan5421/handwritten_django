from django.urls import path
from django.conf.urls import url
from . import views

urlpatterns = [
    path("", views.index, name = "index"),
    path("image_digit/", views.image_digit, name = "image_digit"),
    path('image_alphabet/', views.image_alphabet, name = 'image_alphabet'),
    
    path("canvas_digit/", views.canvas_digit, name = "canvas_digit"),
    #path("digit/" ,views.getimagefromrequest),

]
