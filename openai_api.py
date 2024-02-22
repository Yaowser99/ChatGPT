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



















