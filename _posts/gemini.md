---
layout: post
title: "@title Configure Gemini API key"
---

<a href="https://colab.research.google.com/github/nishzsche/nishzsche.github.io/blob/gh-pages/gemini.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## Gemini: Connecting to Gemini

The Gemini API allows you to connect to Google's most powerful multi-modal model. This example configures your API key and sends an example message to the API and prints a response.

Before you start, visit https://aistudio.google.com/app/apikey to create an API key.


```python
# @title Configure Gemini API key

import google.generativeai as genai
from google.colab import userdata

gemini_api_secret_name = 'GOOGLE_API_KEY'  # @param {type: "string"}

try:
  GOOGLE_API_KEY=userdata.get(gemini_api_secret_name)
  genai.configure(api_key=GOOGLE_API_KEY)
except userdata.SecretNotFoundError as e:
   print(f'Secret not found

This expects you to create a secret named {gemini_api_secret_name} in Colab

Visit https://aistudio.google.com/app/apikey to create an API key

Store that in the secrets section on the left side of the notebook (key icon)

Name the secret {gemini_api_secret_name}')
   raise e
except userdata.NotebookAccessError as e:
  print(f'You need to grant this notebook access to the {gemini_api_secret_name} secret in order for the notebook to access Gemini on your behalf.')
  raise e
except Exception as e:
  print(f"There was an unknown error. Ensure you have a secret {gemini_api_secret_name} stored in Colab and it's a valid key from https://aistudio.google.com/app/apikey")
  raise e
```


```python
# @title Connect to the API and send an example message

text = 'What is the velocity of an unladen swallow?' # @param {type: "string"}

model = genai.GenerativeModel('gemini-pro')
chat = model.start_chat(history=[])

response = chat.send_message(text)
response.text
```

## Gemini: Creating a prompt

This rich example shows how you can create and configure complex prompts for Gemini. It assumes that you've already created an API key at https://aistudio.google.com/app/apikey and added it to your Colab secrets as `GOOGLE_API_KEY` (see the "Connecting to Gemini" snippet).


```python
# @title Create a prompt

import google.generativeai as genai
from google.colab import userdata

api_key_name = 'GOOGLE_API_KEY' # @param {type: "string"}
prompt = 'What is the velocity of an unladen swallow?' # @param {type: "string"}
system_instructions = 'You have a tendency to speak in riddles.' # @param {type: "string"}
model = 'gemini-1.5-flash' # @param {type: "string"} ["gemini-1.0-pro", "gemini-1.5-pro", "gemini-1.5-flash"]
temperature = 0.5 # @param {type: "slider", min: 0, max: 2, step: 0.05}
stop_sequence = '' # @param {type: "string"}

if model == 'gemini-1.0-pro' and system_instructions is not None:
  system_instructions = None
  print('[31m(WARNING: System instructions ignored, gemini-1.0-pro does not support system instructions)[0m')

if model == 'gemini-1.0-pro' and temperature > 1:
  temperature = 1
  print('[34m(INFO: Temperature set to 1, gemini-1.0-pro does not support temperature > 1)[0m')

if system_instructions == '':
  system_instructions = None

api_key = userdata.get(api_key_name)
genai.configure(api_key=api_key)
model = genai.GenerativeModel(model, system_instruction=system_instructions)
config = genai.GenerationConfig(temperature=temperature, stop_sequences=[stop_sequence])
response = model.generate_content(contents=[prompt], generation_config=config)
response.text
```
