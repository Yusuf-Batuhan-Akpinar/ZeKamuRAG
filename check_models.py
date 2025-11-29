
import google.generativeai as genai
import os

api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyAnPBEZG9L19iupfuJVuo-ZJhNOn567BvQ")
genai.configure(api_key=api_key)

print("Available models:")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
