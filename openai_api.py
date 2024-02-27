## Text Completion
# Set your API key
client = OpenAI(api_key="____")

messages = [{"role": "system", "content": "You are a helpful math tutor."}]
user_msgs = ["Explain what pi is.", "Summarize this in two bullet points."]

for q in user_msgs:
    print("User: ", q)
    
    # Create a dictionary for the user message from q and append to messages
    user_dict = {"role": "user", "content": q}
    messages.append(user_dict)
    
    # Create the API request
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100
    )
    
    # Convert the assistant's message to a dict and append to messages
    assistant_dict = {"role": "assistant", "content": response.choices[0].message.content}
    messages.append(assistant_dict)
    print("Assistant: ", response.choices[0].message.content, "\n")





## Text Moderation
# Set your API key
client = OpenAI(api_key="____")

# Create a request to the Moderation endpoint
response = client.moderations.create(
    model="text-moderation-latest",
    input="My favorite book is How to Kill a Mockingbird."
)

# Print the category scores
print(response.results[0].category_scores)





## Audio Transciption
# Set your API key
client = OpenAI(api_key="____")

# Open the openai-audio.mp3 file
audio_file = open("openai-audio.mp3", "rb")

# Create a transcript from the audio file
response = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

# Extract and print the transcript text
print(response.text)



## Audio Translation (other language to english)
# Set your API key
client = OpenAI(api_key="____")

# Open the audio.m4a file
audio_file = open("audio.m4a", "rb")

# Create a translation from the audio file
response = client.audio.translations.create(model="whisper-1", file=audio_file)

# Extract and print the translated text
print(response.text)

# Adding a prompt to assist translation process
# Set your API key
client = OpenAI(api_key="____")

# Open the audio.wav file
audio_file = open("audio.wav","rb")

# Write an appropriate prompt to help the model
prompt = "This audio is related to a recent world bank report"

# Create a translation from the audio file
response = client.audio.translations.create(
    model="whisper-1",
    file=audio_file,
    prompt=prompt
)

print(response.text)


## Combined Models
# Set your API key
client = OpenAI(api_key="____")

# Open the audio.wav file
audio_file = open("audio.wav", "rb")

# Create a transcription request using audio_file
audio_response = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file
)

transcript=audio_response.text
prompt="Determine the language used in the following text: " + transcript

# Create a request to the API to identify the language spoken
chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user","content":prompt}]
)
print(chat_response.choices[0].message.content)

# Set your API key
client = OpenAI(api_key="____")

# Open the datacamp-q2-roadmap.mp3 file
audio_file = open("datacamp-q2-roadmap.mp3","rb")

# Create a transcription request using audio_file
audio_response = client.audio.transcriptions.create(model="whisper-1",file=audio_file)

transcript=audio_response.text
prompt="Summarize the following transcript into bullet points: " + transcript

# Create a request to the API to summarize the transcript into bullet points
chat_response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role":"user","content":prompt}]
)
print(chat_response.choices[0].message.content)


## Embeddings
# Create an OpenAI client and set your API key
client = OpenAI(api_key="____")

# Create a request to obtain embeddings
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input="Embeddings are a numerical representation of text that can be used to measure the relatedness between two pieces of text."
)

# Convert the response into a dictionary
response_dict = response.model_dump()

print(response_dict)
# Extrat the embeddings from the response
print(response_dict['data'][0]['embedding'])
# Extract the total tokens from response
print(response_dict['usage']['total_tokens'])



## Embedding multiple inputs, storing and handling. 
# Set your API key
client = OpenAI(api_key="____")

# Extract a list of product short descriptions from products
product_descriptions = [product['short_description'] for product in products]

# Create embeddings for each product description
response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=product_descriptions
)
response_dict = response.model_dump()

# Extract the embeddings from response_dict and store in products
for i, product in enumerate(products):
    product['embedding'] = response_dict['data'][i]['embedding']
    
print(products[0].items())

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
# Create reviews and embeddings lists using list comprehensions
categories = [product['category'] for product in products]
embeddings = [product['embedding'] for product in products]

# Reduce the number of embeddings dimensions to two using t-SNE
tsne = TSNE(n_components=2, perplexity=5)
embeddings_2d = tsne.fit_transform(np.array(embeddings))

# Create a scatter plot from embeddings_2d
plt.scatter(embeddings_2d[:,0], embeddings_2d[:,1])

for i, category in enumerate(categories):
    plt.annotate(category, (embeddings_2d[i, 0], embeddings_2d[i, 1]))

plt.show()


## Text Similariy
# Set your API key
client = OpenAI(api_key="____")

# Define a create_embeddings function
from scipy.spatial import distance
import numpy as np
def create_embeddings(texts):
  response = client.embeddings.create(
    model="text-embedding-ada-002",
    input=texts
  )
  response_dict = response.model_dump()
  
  return [data['embedding'] for data in response_dict['data']]

# Embed short_description and print
print(create_embeddings(short_description)[0])

# Embed list_of_descriptions and print
print(create_embeddings(list_of_descriptions))
# Set your API key
client = OpenAI(api_key="____")

# Embed the search text
search_text = "soap"
search_embedding = create_embeddings(search_text)[0]

distances = []
for product in products:
  # Compute the cosine distance for each product description
  dist = distance.cosine(search_embedding, product['embedding'])
  distances.append(dist)

# Find and print the most similar product short_description    
min_dist_ind = np.argmin(distances)
print(products[min_dist_ind]['short_description'])


## Semantic search and enriched embeddings
# Set your API key
client = OpenAI(api_key="____")

# Define a function to combine the relevant features into a single string
def create_product_text(product):
  return f"""Title: {product['title']}
Description: {product['short_description']}
Category: {product['category']}
Features: {'; '.join(product['features'])}"""

# Combine the features for each product
product_texts = [create_product_text(product) for product in products]

# Create the embeddings from product_texts
product_embeddings = create_embeddings(product_texts)

def find_n_closest(query_vector, embeddings, n=3):
  distances = []
  for index, embedding in enumerate(embeddings):
    # Calculate the cosine distance between the query vector and embedding
    dist = distance.cosine(query_vector, embedding)
    # Append the distance and index to distances
    distances.append({"distance": dist, "index": index})
  # Sort distances by the distance key
  distances_sorted = sorted(distances, key=lambda x:x['distance'])
  # Return the first n elements in distances_sorted
  return distances_sorted[0:n]

# Set your API key
client = OpenAI(api_key="____")

# Create the query vector from query_text
query_text = "computer"
query_vector = create_embeddings(query_text)[0]

# Find the five closest distances
hits = find_n_closest(query_vector, product_embeddings, n=5)

print(f'Search results for "{query_text}"')
for hit in hits:
  # Extract the product at each index in hits
  product = products[hit['index']]
  print(product["title"])






























