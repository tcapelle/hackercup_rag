import openai
import os
import json

print("Trying OpenAI client with json.loads")
client = openai.OpenAI(base_url="https://api.mistral.ai/v1", 
                       api_key=os.environ["MISTRAL_API_KEY"])

prompt = "WHat is the capital of France, reply in JSON country:capital"

res = client.chat.completions.create(
    model="open-mistral-nemo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"}, 
        {"role": "user", "content": prompt}],
    response_format={ "type": "json_object" },
    max_tokens=64,
).choices[0].message.content

try:
    generation = json.loads(res)
except:
    generation = {}

print(generation)
print("*"*100)
print("Trying OpenAI client with instructor")


from pydantic import BaseModel
import instructor
instructor_client = instructor.from_openai(client)


class City(BaseModel):
    country: str
    capital: str

res = instructor_client.chat.completions.create(
    model="open-mistral-nemo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant"}, 
        {"role": "user", "content": prompt}],
    response_model=City,
    max_tokens=64,
)


print(res)
print("*"*100)
