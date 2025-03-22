from django.contrib import admin
from django.urls import path
from base import views
from django.conf import settings
from django.conf.urls.static import static
urlpatterns = [
    path('admin/', admin.site.urls),
    path('',views.HomePage,name='home'),
    path('signup/',views.SignupPage,name='signup'),
    path('index/',views.index,name='index'),
    path('index2/',views.index2,name='index2'),
    path('index3/',views.index3,name='index3'),
    path('index4/',views.index4,name='index4'),
    path('chatbot/', views.chatbot_view, name='chatbot_view'),
    path('login/',views.LoginPage,name='login'),
    path('logout/',views.LogoutPage,name='logout'),
    path('index/result/', views.result, name='result'),
    path('index2/result2/', views.result2, name='result2'),
    path('index3/result3/', views.result3, name='result3'),
    
]