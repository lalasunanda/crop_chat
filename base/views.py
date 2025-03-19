import os
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse, HttpResponseBadRequest
from django.contrib.auth.models import User
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.views.decorators.csrf import csrf_exempt
import json
import random
import numpy as np
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
import requests
from datetime import datetime, timezone, timedelta
from django.contrib import messages

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

lemmatizer = WordNetLemmatizer()

# Define BASE_DIR to get correct paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load models and data
try:
    tfidf_fit = joblib.load(os.path.join(BASE_DIR, 'Models', 'tfidf_vectorizer.pkl'))
    tfidf_corpus = joblib.load(os.path.join(BASE_DIR, 'Models', 'tfidf_corpus.pkl'))
    df = pd.read_csv(os.path.join(BASE_DIR, 'Models', 'questions_answers.csv'))
except FileNotFoundError as e:
    print(f"File loading error: {e}")

def clean_data(text):
    text = re.sub(r"[\([{})\]]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)

def HomePage(request):
    return render(request, 'home.html')

def is_valid_password(password):
    return (
        len(password) >= 8 and
        re.search(r'[A-Z]', password) and
        re.search(r'[a-z]', password) and
        re.search(r'\d', password) and
        re.search(r'[!@#$%^&*(),.?":{}|<>]', password)
    )

def SignupPage(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")
        
        has_errors = False
        
        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            has_errors = True
        
        if not is_valid_password(password1):
            messages.error(request, "Password must meet complexity requirements.")
            has_errors = True
        
        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken!")
            has_errors = True
        
        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
            has_errors = True
        
        if not has_errors:
            User.objects.create_user(username=username, email=email, password=password1)
            messages.success(request, "Signup successful! Please log in.")
            return redirect("login")
    
    return render(request, "signup.html")

def LoginPage(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        pass1 = request.POST.get('pass')
        
        user = authenticate(request, username=username, password=pass1)
        if user is not None:
            login(request, user)
            return redirect('index')
        else:
            messages.error(request, "Username or Password is incorrect!")
            return redirect('login')
    return render(request, 'login.html')

def LogoutPage(request):
    logout(request)
    return redirect('home')

def getPredictions(a,b,c,d,e,f,g,h,i):
    model = pickle.load(open(os.path.join(BASE_DIR, 'Models', 'RF.pkl'), 'rb'))
    new_data = {'Crop': a, 'Crop_Year': b, 'Season': c, 'State': d, 'Area': e, 'Production': f, 'Annual_Rainfall': g, 'Fertilizer': h, 'Pesticide': i}
    new_df = pd.DataFrame([new_data])
    prediction = model.predict(new_df)
    return round(prediction[0], 2)

@login_required(login_url='login')
def result(request):
    result = getPredictions(
        request.GET['Crop'], int(request.GET['Crop_Year']), request.GET['Season'],
        request.GET['State'], int(request.GET['Area']), int(request.GET['Production']),
        float(request.GET['Annual_Rainfall']), float(request.GET['Fertilizer']), float(request.GET['Pesticide'])
    )
    return render(request, 'result.html', {'result': result})

@login_required(login_url='login')
@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        try:
            data = json.loads(request.body.decode('utf-8'))
            user_input = data.get('user_input', '').strip()
            if not user_input:
                return JsonResponse({'error': 'Empty input'}, status=400)
            
            GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey")
            GREETING_RESPONSES = ["hi there", "hello", "Hi, I am glad! You are talking to me"]
            
            if any(word in user_input.lower() for word in GREETING_INPUTS):
                response_text = random.choice(GREETING_RESPONSES).capitalize()
            elif user_input in ['thanks', 'thank you']:
                response_text = "You are welcome."
            elif user_input == 'what is your name?':
                response_text = "I am a chatbot."
            elif user_input == 'bye':
                response_text = "Bye! Take care."
            else:
                user_input_cleaned = clean_data(user_input)
                tfidf_test = tfidf_fit.transform([user_input_cleaned])
                cosine_similarities = cosine_similarity(tfidf_test, tfidf_corpus).flatten()
                highest_similarity_index = cosine_similarities.argmax()
                if cosine_similarities[highest_similarity_index] == 0:
                    response_text = "I'm sorry, I don't have an answer for that."
                else:
                    response_text = df.iloc[highest_similarity_index]['answers'].capitalize()
        except Exception as e:
            print(f"Error: {e}")
            return JsonResponse({'error': 'An error occurred'}, status=500)
        return JsonResponse({'response': response_text})
    return HttpResponseBadRequest("Only POST allowed.")

@login_required(login_url='login')
def index(request):
    return render(request, 'index.html')



