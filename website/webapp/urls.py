from django.urls import path

from . import views

urlpatterns = [
    path("", views.index_, name="index_"),
    path("home", views.index, name='index')
]