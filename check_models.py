
import google.generativeai as genai
import os

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("Hata: GOOGLE_API_KEY bulunamadı. Lütfen environment variable olarak tanımlayın.")
    exit(1)
genai.configure(api_key=api_key)

print("Available models:")
for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)
