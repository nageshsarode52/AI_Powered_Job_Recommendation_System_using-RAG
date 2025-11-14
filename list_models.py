# list_models.py
import os
import google.generativeai as genai

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

models = list(genai.list_models())

print("\nAVAILABLE MODELS:\n")
for m in models:
    print("MODEL NAME:", getattr(m, "name", None))
    print("DISPLAY NAME:", getattr(m, "display_name", None))
    print("SUPPORTED METHODS:", getattr(m, "supported_methods", None))
    print("-" * 50)