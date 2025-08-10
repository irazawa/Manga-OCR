import deepl
import google.generativeai as genai

auth_key = "2107028d-26f0-4604-a625-80d62f226311:fx"
translator = deepl.Translator(auth_key)

result = translator.translate_text("Hello, world!", target_lang="ID")
print(result.text)

genai.configure(api_key="AIzaSyCRJNB4SukhoZvOaZNCO9p1bLCNMnHABTM")

for m in genai.list_models():
    if "generateContent" in m.supported_generation_methods:
        print(m.name)
