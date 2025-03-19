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
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


lemmatizer = WordNetLemmatizer()

# Load TF-IDF model, corpus, and dataframe
try:
    tfidf_fit = joblib.load('Models\\tfidf_vectorizer.pkl')
    tfidf_corpus = joblib.load('Models\\tfidf_corpus.pkl')
    df = pd.read_csv('Models\\questions_answers.csv')
except FileNotFoundError as e:
    print(f"File loading error: {e}")





# Utility function to clean and preprocess text
def clean_data(text):
    text = re.sub(r"[\([{})\]]", " ", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return " ".join(tokens)


# Home page view
def HomePage(request):
    return render(request, 'home.html')


# Signup view
from django.shortcuts import render, redirect
from django.contrib.auth.models import User
from django.contrib import messages


def is_valid_password(password):
    """Check if password meets complexity requirements."""
    return (
        len(password) >= 8 and
        re.search(r'[A-Z]', password) and    # At least one uppercase letter
        re.search(r'[a-z]', password) and    # At least one lowercase letter
        re.search(r'\d', password) and       # At least one number
        re.search(r'[!@#$%^&*(),.?":{}|<>]', password)  # At least one special character
    )

def SignupPage(request):
    if request.method == "POST":
        username = request.POST.get("username")
        email = request.POST.get("email")
        password1 = request.POST.get("password1")
        password2 = request.POST.get("password2")

        has_errors = False  # Track errors

        if password1 != password2:
            messages.error(request, "Passwords do not match!")
            has_errors = True

        if not is_valid_password(password1):
            messages.error(request, "Password must be at least 8 characters long and include an uppercase letter, a lowercase letter, a number, and a special character.")
            has_errors = True


        if User.objects.filter(username=username).exists():
            messages.error(request, "Username already taken!")
            has_errors = True

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already registered!")
            has_errors = True

        if not has_errors:  # Proceed only if no errors
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
            return redirect('login')  # Redirect back to login page with error message

    return render(request, 'login.html')


# Logout view
def LogoutPage(request):
    logout(request)
    return redirect('home')

def getPredictions(a,b,c,d,e,f,g,h,i):
    model = pickle.load(open('Models\\RF.pkl', 'rb'))
    new_data = {'Crop': a,
            'Crop_Year': b,
            'Season': c,
            'State': d,
            'Area': e,
            'Production': f,
            'Annual_Rainfall': g,
            'Fertilizer': h,
            'Pesticide': i
           }
    new_df = pd.DataFrame([new_data])
    prediction = model.predict(new_df)
    return prediction[0]

@login_required(login_url='login')
def result(request):
    a = str(request.GET['Crop'])
    b = int(request.GET['Crop_Year'])
    c = str(request.GET['Season'])
    d = str(request.GET['State'])
    e = int(request.GET['Area'])
    f = int(request.GET['Production'])
    g = float(request.GET['Annual_Rainfall'])
    h = float(request.GET['Fertilizer'])
    i = float(request.GET['Pesticide'])
    result = getPredictions(a, b, c, d, e, f, g,h,i)
    result=round(result,2)
    return render(request, 'result.html', {'result': result})

def getPredictions2(a, b, c, d, e, f):
    model = pickle.load(open('Models\\Crop_season_model.pkl', 'rb'))
    new_data = {
            'State_Name': a,
            'District_Name':b,
           'Crop_Year':c,
            'Season': d,
           'Area':e,
            'Production':f
           }
    new_df = pd.DataFrame([new_data])
    prediction = model.predict(new_df)

    return prediction[0]

@login_required(login_url='login')
def result2(request):
    a = str(request.GET['State_Name'])
    b = str(request.GET['District_Name'])
    c = int(request.GET['Crop_Year'])
    d = str(request.GET['Season'])
    e = float(request.GET['Area'])
    f = float(request.GET['Production'])
    result = getPredictions2(a, b, c, d, e, f)
    return render(request, 'result2.html', {'result': result})


@login_required(login_url='login')
def result3(request):
    city = str(request.GET['city'])
    api_key = "ef4a754ca8160dc1940f49c6546c1c00"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    data = requests.get(url).json()
    try:
        if data['cod'] == '404':
            return HttpResponse('{"status": "notfound"}')
        else:
            city_name = data['name']
            country = data.get('sys').get('country', '-')
            ts = data['dt']
            tzone = data['timezone']
            date_time = datetime.fromtimestamp(ts, tz=timezone(timedelta(seconds=tzone))).strftime('%Y-%m-%d')
            temp = int(data['main']['temp'])
            temp_F = format((temp*1.8)+32, '.1f')
            description = data['weather'][0]['description']
            humidity = data['main']['humidity']
            feels_like = int(data['main']['feels_like'])
            wind = format(data['wind']['speed']*3.6, '.1f')
            visibility = format(data['visibility']/1000, '.2f')
    except:
            return HttpResponse('{"status": "error"}')
    
    return render(request, 'result3.html', {'city': city_name, 'country': country, 'date_time':date_time, 'temp': temp, 'temp_F': temp_F, 'description': description, 'humidity': humidity, 'feels_like': feels_like, 'wind': wind, 'visibility': visibility})









@login_required(login_url='login')
@csrf_exempt
def chatbot_view(request):
    if request.method == 'POST':
        try:
            # Decode the JSON data from the request body
            data = json.loads(request.body.decode('utf-8'))
            print("Received data:", data)  # Log incoming data

            # Check if the 'user_input' field is present
            user_input = data.get('user_input', '').strip()
            if not user_input:
                return JsonResponse({'error': 'Missing or empty "user_input" field'}, status=400)

            # Define greeting responses
            GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey",)
            GREETING_RESPONSES = ["hi there", "hello", "Hi, I am glad! You are talking to me"]

            # Check for greetings
            if any(word in user_input.lower() for word in GREETING_INPUTS):
                response_text = random.choice(GREETING_RESPONSES).capitalize()
            elif user_input in ['thanks', 'thank you']:
                response_text = "You are welcome."
            elif user_input == 'what is your name?':
                response_text = "I am a chatbot."
            elif user_input == 'bye':
                response_text = "Bye! Take care."
            else:
                # Clean and process the user input
                user_input_cleaned = clean_data(user_input)

                # Transform the user input into TF-IDF features
                tfidf_test = tfidf_fit.transform([user_input_cleaned])
                cosine_similarities = cosine_similarity(tfidf_test, tfidf_corpus).flatten()

                # Find the highest similarity index
                highest_similarity_index = cosine_similarities.argmax()

                # Check if there is a meaningful match
                if cosine_similarities[highest_similarity_index] == 0:
                    response_text = "I'm sorry, I don't have an answer for that."
                else:
                    # Get the answer from the dataframe
                    response_text = df.iloc[highest_similarity_index]['answers'].capitalize()

        except Exception as e:
            # Handle any unexpected errors
            print(f"Error processing request: {e}")
            return JsonResponse({'error': 'An unexpected error occurred'}, status=500)

        # Return the chatbot response as JSON
        return JsonResponse({'response': response_text})

    # Return BadRequest if the method is not POST
    return HttpResponseBadRequest("Only POST method is allowed.")
# Additional views
@login_required(login_url='login')
def index(request):
    return render(request, 'index.html')

@login_required(login_url='login')
def index2(request):
    return render(request, 'index2.html')
@login_required(login_url='login')
def index3(request):
    return render(request, 'index3.html')
@login_required(login_url='login')
def index4(request):
    return render(request, 'chatbot.html')