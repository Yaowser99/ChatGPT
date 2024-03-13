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



# Recommendation Systems
# Set your API key
client = OpenAI(api_key="____")

# Combine the features for last_product and each product in products
last_product_text = create_product_text(last_product)
product_texts = [create_product_text(product) for product in products]

# Embed last_product_text and product_texts
last_product_embeddings = create_embeddings(last_product_text)[0]
product_embeddings = create_embeddings(product_texts)

# Find the three smallest cosine distances and their indexes
hits = find_n_closest(last_product_embeddings, product_embeddings)

for hit in hits:
  product = products[hit['index']]
  print(product['title'])

# Set your API key
client = OpenAI(api_key="____")

# Prepare and embed the user_history, and calculate the mean embeddings
history_texts = [create_product_text(product) for product in user_history]
history_embeddings = create_embeddings(history_texts)
mean_history_embeddings = np.mean(history_embeddings, axis=0)

# Filter products to remove any in user_history
products_filtered = [product for product in products if product not in user_history]

# Combine product features and embed the resulting texts
product_texts = [create_product_text(product) for product in products_filtered]
product_embeddings = create_embeddings(product_texts)

hits = find_n_closest(mean_history_embeddings, product_embeddings)

for hit in hits:
  product = products_filtered[hit['index']]
  print(product['title'])



# Embeddings for classifications
sentiments = [{'label': 'Positive'},
              {'label': 'Neutral'},
              {'label': 'Negative'}]

reviews = ["The food was delicious!",
           "The service was a bit slow but the food was good",
           "Never going back!"]

# Set your API key
client = OpenAI(api_key="____")

# Create a list of class descriptions from the sentiment labels
class_descriptions = [sentiment['label'] for sentiment in sentiments]

# Embed the class_descriptions and reviews
class_embeddings = create_embeddings(class_descriptions)
review_embeddings = create_embeddings(reviews)[0]

# Define a function to return the minimum distance and its index
def find_closest(query_vector, embeddings):
  distances = []
  for index, embedding in enumerate(embeddings):
    dist = distance.cosine(query_vector, embedding)
    distances.append({"distance": dist, "index": index})
  return min(distances, key=lambda x: x["distance"])

for index, review in enumerate(reviews):
  # Find the closest distance and its index using find_closest()
  closest = find_closest(review_embeddings[index], class_embeddings)
  # Subset sentiments using the index from closest
  label = sentiments[closest['index']]['label']
  print(f'"{review}" was classified as {label}')

# Set your API key
client = OpenAI(api_key="____")

# Extract and embed the descriptions from sentiments
class_descriptions = [sentiment['description'] for sentiment in sentiments]
class_embeddings = create_embeddings(class_descriptions)
review_embeddings = create_embeddings(reviews)

def find_closest(query_vector, embeddings):
  distances = []
  for index, embedding in enumerate(embeddings):
    dist = distance.cosine(query_vector, embedding)
    distances.append({"distance": dist, "index": index})
  return min(distances, key=lambda x: x["distance"])

for index, review in enumerate(reviews):
  closest = find_closest(review_embeddings[index], class_embeddings)
  label = sentiments[closest['index']]['label']
  print(f'"{review}" was classified as {label}')



## Vector Databases for embedding systems
import chromadb
# Create a persistant client
client = chromadb.PersistentClient()

from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
# Create a netflix_title collection using the OpenAI Embedding function
collection = client.create_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(api_key="____")
)

# List the collections
print(client.list_collections())
# pip install tiktoken

import tiktoken
# Load the encoder for the OpenAI text-embedding-ada-002 model
enc = tiktoken.encoding_for_model("text-embedding-ada-002")

# Encode each text in documents and calculate the total tokens
total_tokens = sum(len(enc.encode(text)) for text in documents)

cost_per_1k_tokens = 0.0001

# Display number of tokens and cost
print('Total tokens:', total_tokens)
print('Cost:', cost_per_1k_tokens*total_tokens/1000)

# Recreate the netflix_titles collection
collection = client.create_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(api_key="____")
)

# Add the documents and IDs to the collection
ids = []
documents = []

with open('netflix_titles.csv') as csvfile:
  reader = csv.DictReader(csvfile)
  for i, row in enumerate(reader):
    ids.append(row['show_id'])
    text = f"Title: {row['title']} ({row['type']})\nDescription: {row['description']}\nCategories: {row['listed_in']}"
    documents.append(text)

collection.add(
  ids = ids,
  documents=documents
)

# Print the collection size and first ten items
print(f"No. of documents: {collection.count()}")
print(f"First ten documents: {collection.peek()}")



## Querying and updating the database
# Retrieve the netflix_titles collection
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(api_key="____")
)

# Query the collection for "films about dogs"
result = collection.query(
  query_texts=['films about dogs'],
  n_results=3
)

print(result)

# Retrieve the netflix_titles collection
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(api_key="____")
)

# Update or add the new documents
collection.upsert(
  ids=[doc['id'] for doc in new_data],
  documents=[doc['document'] for doc in new_data]
)

# Delete the item with ID "s95" and re-run the query
collection.delete(ids=["s95"])

result = collection.query(
  query_texts=["films about dogs"],
  n_results=3
)
print(result)



# Multiple queries and filtering
# Retrieve the netflix_titles collection
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(api_key="____")
)

reference_ids = ['s999', 's1000']

# Retrieve the documents for the reference_ids
reference_texts = collection.get(ids=reference_ids)['documents']

# Query using reference_texts
result = collection.query(
  query_texts=reference_texts,
  n_results=3
)

print(result['documents'])

# Retrieve the netflix_titles collection
collection = client.get_collection(
  name="netflix_titles",
  embedding_function=OpenAIEmbeddingFunction(api_key="____")
)

reference_texts = ["children's story about a car", "lions"]

# Query two results using reference_texts
result = collection.query(
  query_texts=reference_texts,
  n_results=2,
  # Filter for titles with a G rating released before 2019
  where={
    "$and": [
        {"rating": 
        	{"$eq":"G"}
        },
        {"release_year": 
         	{"$lt":2019}
        }
    ]
  }
)

print(result['documents'])



# Introduction to Prompt Engineering 
# Set your API key
client = OpenAI(api_key="____")

def get_response(prompt):
  # Create a request to the chat completions endpoint
  response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    # Assign the role and content for the message
    messages=[{"role": "user", "content": prompt}], 
    temperature = 0)
  return response.choices[0].message.content

# Test the function with your prompt
response = get_response("write a poem about ChatGPT")
print(response)

# Craft a prompt that follows the instructions
prompt = "write a poem about ChatGPT in basic English that a child can understand"

# Get the response
response = get_response(prompt)

print(response)



# Key Principles of prompt Engineering
# Create a prompt that completes the story (story variable stores the given story)
prompt = f"""Complete the given story delimited by triple backticks.
          ```{story}```"""

# Get the generated response 
response = get_response(prompt)

print("\n Original story: \n", story)
print("\n Generated story: \n", response)

# Create a request to complete the story
prompt = f"""Complete the given story delimited by triple backticks. Do so with only two paragraphs and in the style of William Shakespeare.
```{story}```"""

# Get the generated response
response = get_response(prompt)

print("\n Original story: \n", story)
print("\n Generated story: \n", response)


# Structured outputs and conditional prompts
# Create a prompt that generates the table
prompt = "Generate a table of 10 books with columns for Title, Author, and Year that I should read as a sci-fi lover. "

# Get the response
response = get_response(prompt)
print(response)


# Create the instructions
instructions = "Define the language and generate a suitable title for the text delimited by triple backticks."

# Create the output format
output_format = """Use the following format for the output: 
    - Text: <The text we used>
    - Language: <The languag we defined>
    - Title: <The title we generated>
"""

# Create the final prompt
prompt = instructions + output_format + f"{text}"
response = get_response(prompt)
print(response)

# Create the instructions
instructions = "You will infer the language and the number of sentences of the given text delimited with triple backticks. If the text contains more than one sentence, generate a suitable title for it, otherwise, write 'N/A' for the title. "

# Create the output format
output_format = """Use the given format for the output: 
    - Text: <The text we used>
    - Language: <The language we inferred>
    - N_sentences: <The sentences we inferred>
    - Title: <The title we generated>
"""

prompt = instructions + output_format + f"```{text}```"
response = get_response(prompt)
print(response)



## Few shot Prompting
# Create a one-shot prompt
prompt = """
Q: Extract the odd numbers from the set {1,3,7,12,19}. A: {1,3,7,19}
Q: Extract the odd numbers from the set {3,5,11,12,16}. A: 
"""

response = get_response(prompt)
print(response)


response = client.chat.completions.create(
  model = "gpt-3.5-turbo",
  # Provide the examples as previous conversations
  messages = [{"role": "user", 
               "content": "The product quality exceeded my expectations"},
              {"role": "assistant",
               "content": "1"},
              {"role": "user",
               "content": "I had a terrible experience with this product's customer service"},
              {"role": "assistant",
               "content": "-1"},
              # Provide the text for the model to classify
              {"role": "user",
               "content": "The price of the product is really fair given its features"}
             ],
  temperature = 0
)
print(response.choices[0].message.content)



# Multistep prompting
# Create a prompt detailing steps to plan the trip
prompt = """
     Help me plan a beach vacation.
     Step 1 - Specify four potential locations for beach vacations
     Step 2 - State some accommodation options in each
     Step 3 - State activities that could be done in each
     Step 4 - Evaluate the pros and cons for each destination
    """

response = get_response(prompt)
print(response)


code = '''
def calculate_rectangle_area(length, width):
    area = length * width
    return area
'''

# Create a prompt that analyzes correctness of the code
prompt = f"""
Please assess the code function delimited by triple backticks. Follow the steps below: 
Step 1: Evaluate if the function has correct syntax
Step 2: Check if the function receives two inputs
Step 3: Check if hte function returns one output. 
```{code}```
"""

response = get_response(prompt)
print(response)



## Chain of thought and self consistency promption
# Set your API key
client = OpenAI(api_key="")

# Create the chain-of-thought prompt
prompt = """Q: Your friend is 20 year old. His father is now double of your friend's age. What is your friend's father's age in 10 years? A: Let's think step by step"""

response = get_response(prompt)
print(response)


# Define the example 
example = """Q: Sum the even numbers in the following set: {9, 10, 13, 4, 2}.
             A: Even numbers: {10,4,2}. Adding them: 10+4+2=16"""

# Define the question
question = """Q: Sum the even numbers in the following set: {15, 13, 82, 7, 14}. 
              A:"""

# Create the final prompt
prompt = example+question
response = get_response(prompt)
print(response)

# Create the self_consistency instruction
self_consistency_instruction = "Imagine three completely independent experts who reason differently are answering this questions. The final answer is obtained by majority vote. The question is: "

# Create the problem to solve
problem_to_solve = "If you own a store that sells laptops and mobile phones. You start your day with 50 devices in the store, out of which 60% are mobile phones. Throughout the day, three clients visited the store, each of them bought one mobile phone, and one of them bought additionally a laptop. Also, you added to your collection 10 laptops and 5 mobile phones. How many laptops and mobile phones do you have by the end of the day?"

# Create the final prompt
prompt = self_consistency_instruction + problem_to_solve

response = get_response(prompt)
print(response)


# Iterative prompt engineering and refindment
# Refine the following prompt
prompt = "Give me the top 10 pre-trained language models in a table. It should include model name, release year, and owning company. "

response = get_response(prompt)
print(response)


# Refine the following prompt
prompt = """
Receiving a promotion at work made me feel on top of the world -> Happiness
The movie's ending left me with a heavy feeling in my chest -> Sadness
Walking alone in the dark alley sent shivers down my spine -> Fear
Time flies in the modern society with the rapid development of technology -> no explicit emotion
____
Time flies like an arrow ->
"""

response = get_response(prompt)
print(response)


# Text summarization and expansion
# Craft a prompt to summarize the report
prompt = f"""Summarize the report dlimited by triple backticks, in maximum five sentences, focusing on aspects related to AI and data privacy: 
```{report}``` """

response = get_response(prompt)

print("Summarized report: \n", response)


# Craft a prompt to summarize the product description
prompt = f"""Summarize the product description delimited by triple backticks in no more than five bullet points, focusing on the product features and specifications: 
```{product_description}```"""

response = get_response(prompt)

print("Original description: \n", product_description)
print("Summarized description: \n", response)


# Craft a prompt to expand the product's description
prompt = f"""Expend the product description delimited by triple backticks to provide a comprehensive overview capturing the product's features, benefits and potential applications. Limit the response to one paragraph. 
```{product_description}```
"""

response = get_response(prompt)

print("Original description: \n", product_description)
print("Expanded description: \n", response)



#Text Tranormation
# Craft a prompt that translates
prompt = f"""Translate the marketing message delimited in triple backticks from english to french, spanish, and japanese: 
```{marketing_message}```"""
 
response = get_response(prompt)

print("English:", marketing_message)
print(response)


# Craft a prompt to change the email's tone
prompt = f"""Here is an email delimited by triple backticks, change the tone of this email to professional, positive, and user-centric. 
```{sample_email}```"""

response = get_response(prompt)

print("Before transformation: \n", sample_email)
print("After transformation: \n", response)

# Craft a prompt to transform the text
prompt = f"""Follow the steps to transform the text delimited by triple backticks. Step 1: Proofread the text without changing its structure. Step 2: Change the tone of the text to be formal and friendly. 
```{text}```"""

response = get_response(prompt)

print("Before transformation:\n", text)
print("After transformation:\n", response)



# Text analysis
# Craft a prompt to classify the ticket
prompt = f"""The ticket string is delimited by triple backticks, classify the tircket as technical issue, billing Inquiry, or product feedback. 
```{ticket}```"""

response = get_response(prompt)

print("Ticket: ", ticket)
print("Class: ", response)


# Craft a few-shot prompt to get the ticket's entities
prompt = f"""Ticket: {ticket_1} -> Entities: {entities_1}
             Ticket: {ticket_2} -> Entities: {entities_2}
             Ticket: {ticket_3} -> Entities: {entities_3}
             Ticket: {ticket_4} -> Entities: """

response = get_response(prompt)

print("Ticket: \n", ticket_4)
print("Entities: \n", response)



# Code generation and explantion

# Craft a prompt that asks the model for the function
prompt = """Write a Python function that receives a list of 12 floats representing monthly sales data as input and returns the month with the highest sales value as output. """

response = get_response(prompt)
print(response)

examples="""input = [10, 5, 8] -> output = 24
input = [5, 2, 4] -> output = 12
input = [2, 1, 3] -> output = 7
input = [8, 4, 6] -> output = 19
"""

# Craft a prompt that asks the model for the function
prompt = f"""You are provided wiht input-output examples delimited by triple backticks for a python program that receives a list of numerical values and output estimated completion time. Write a Python function base on the examples. 
```{examples}```"""

response = get_response(prompt)
print(response)


function = """def calculate_area_rectangular_floor(width, length):
					return width*length"""

# Craft a multi-step prompt that asks the model to adjust the function
prompt = f"""You are given a function delimited by triple backticks. This function currently calculates the area of a rectangular floor given its width and length. You need to modify this function with the following steps. 
Step 1: Modify function to return the perimeter of the rectangle as well. 
Step 2: Test if the Inputs or dimensions are positive, if not, display an appropriate error message. 
```{function}```
"""

response = get_response(prompt)
print(response)


# Craft a chain-of-thought prompt that asks the model to explain what the function does
prompt = f"""Explain the Python function delimited by triple backticks. Let's think step by step: 
```{function}```
"""
 
response = get_response(prompt)
print(response)



# Prompt Engineering for Chatbot development

def get_response(system_prompt, user_prompt):
  # Assign the role and content for each message
  messages = [{"role": "system", "content": system_prompt},
      		  {"role": "user", "content": user_prompt}]  
  response = client.chat.completions.create(
      model="gpt-3.5-turbo", messages= messages, temperature=0)
  
  return response.choices[0].message.content

# Try the function with a system and user prompts of your choice
system_prompt="You are a experienced financial analyst that provides investment advice. "
user_prompt="What is an ideal asset allocation strategy in a high interest rate environment? " 
response = get_response(system_prompt, user_prompt)
print(response)


# Define the purpose of the chatbot
chatbot_purpose = "You are a customer support agent supporting an e-commerce company specialized in electronics. You will assist users with inquiries, order tracking, and troubleshooting common issues. "

# Define audience guidelines
audience_guidelines = "The target audience is tech-savvy individuals interested in purchasing electronic gadgets. "

# Define tone guidelines
tone_guidelines = "You should use a professional and user-friendly tone while interacting with customers. "

system_prompt = chatbot_purpose + audience_guidelines + tone_guidelines
response = get_response(system_prompt, "My new headphones aren't connecting to my device")
print(response)

# Define the order number condition
order_number_condition = "If the user is asking about an order without providing an order number, ask the user to provide an order number. "

# Define the technical issue condition
technical_issue_condition = "If the user reports a technical issue, expre empathy by starting the response with 'I'm sorry to hear about your issue with ...'"

# Create the refined system prompt
refined_system_prompt = order_number_condition+technical_issue_condition

response_1 = get_response(refined_system_prompt, "My laptop screen is flickering. What should I do?")
response_2 = get_response(refined_system_prompt, "Can you help me track my recent order?")

print("Response 1: ", response_1)
print("Response 2: ", response_2)








