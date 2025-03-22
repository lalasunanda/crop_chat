from django.urls import path
from . import views

urlpatterns = [
    path('home',views.home,name="home"),
    path('index', views.index, name='index'),
    path('index2', views.index2, name='index2'),
    path('weather/', views.index3, name='index3'),
    path('index4', views.index3, name='index4'),
    path('chatbot', views.chatbot_view, name='chatbot_view'),
    path('signup/',views.SignupPage,name='signup'),
    path('result', views.result, name='result'),
    path('result2', views.result2, name='result2'),
    path('result3', views.result3, name='result3'),
    
    
]