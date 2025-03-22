import os

# Ensure model is downloaded
os.system("python download_models.py")

# Run Django server
os.system("python manage.py runserver 0.0.0.0:7860")
