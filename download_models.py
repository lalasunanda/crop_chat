import os
import gdown

BASE_DIR = "/opt/render/project/src"

# Define the Google Drive file IDs and target paths
files_to_download = {
    "Chatbot/tfidf_corpus.pkl": "1ET0j4J63g5xtK_DpELnG6I-hzKdEOu0i",  # Replace with actual file ID
    "Chatbot/tfid_vectorizer.pkl": "14MuYnZS6GmfkPnqcxo_RNpRfF-Q8GgFB",
    "Chatbot/questions4.csv":"1ngjhJ8zmiU8Sl5XzSfuqVd33UQegIvPW",
    "Chatbot/questions_answers.csv":"1tS-TxWqH3HTonV9EqJXLZW6tf2kWpDmH",
    "Chatbot/A_Chat.ipynb": "1BhDvQALFKlAwc1UcYVdHUv6O3cigKgBn",
    "Crop/Crop.ipynb": "109SpsTFzOUSQBNRFyC7C6nlcbxBl8jGw",
    "Crop/Crop.csv": "16dPVItPRlHu0xJeN7LTjpcFzp4Vq_j-N",
    "Crop season/Crop_season_model.pkl": "1HXqJ_VDCj9wuOvG0nXmOB_LIxagBzQMR",
    "Crop season/Crop_season.ipynb": "1X6fnGsErc17yMXTgFqbX3vATyPIbNsX9",
    "Crop season/Data.csv": "1Uaw6RNX2Y4uXKALid5NcIiD9uVk7i28-",
    "base/Models/Crop_season_model_quantized.onnx": "1JtQiL-9a5cmzwVXh3FO0Ua_OCVEiajze",
    "base/Models/RF.pkl": "117J9IcPutGhGNENHyJGaMRHu73rkxkGh",
    "Models/Weather_Rf.pkl": "1cieV2kNFutthpt1eBo0NBf8FlhzwGNbA",
    "Models/questions_answers.csv": "1V1y98XAwDWSw7vgvBLkHwqHswLrV4cfk",
    "Models/tfidf_corpus.pkl": "1thKDeqZawyz5oDKLiCviE9M4hjnFxvbq",
    "Models/tfidf_vectorizer.pkl": "1AlvfkXVWdk5s8yrQMgMs69-DlF4OTRRz"
}

# Ensure directories exist
for path in files_to_download.keys():
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Download each file
for file_path, file_id in files_to_download.items():
    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"Downloading {file_path}...")
    gdown.download(url, file_path, quiet=False)

print("All files downloaded successfully!")
