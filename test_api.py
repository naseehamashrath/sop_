from google import genai

client = genai.Client(api_key="PASTE_YOUR_FULL_API_KEY_HERE")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="Say hello."
)

print(response.text)