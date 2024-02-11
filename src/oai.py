import base64

from openai import OpenAI
import openai
import json

client = OpenAI(api_key='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')

# Function to encode the image
def text_desc(image_file):
    # Getting the base64 string
    base64_image =  base64.b64encode(image_file.read()).decode('utf-8')
    completion = client.chat.completions.create(
    model =  "gpt-4-vision-preview",
    messages= [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Give a breif description of the clothing article in the image"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    max_tokens = 300
    )
    

    # print(completion.choices[0].message)

            
    return completion.choices[0].message.content


def suggest_outfit(event, dress_string):
    # Assuming `openai` is already configured with your API key
    # Split the dress_string into a list
    clothes_list = dress_string.split(',')
    
    # Construct a detailed prompt for GPT-4
    prompt = (
        f"I have the following items in my wardrobe: {', '.join(clothes_list)}. "
        f"I am attending {event}. Based on these items, "
        "what would be the perfect outfit combination for the event? "
        "Provide a suggestion with a description, followed by '--TDTM--' and "
        "a structured JSON detailing each item of the outfit."
        "reduce the main message to 300 tokens."
    )
    
    # Get the completion from GPT-4
    completion = client.chat.completions.create(
    model =  "gpt-4-turbo-preview",
    messages= [
        {"role": "system","content": "You are a fashion assistant."},
        { "role": "user","content": prompt}
            ],
        
    max_tokens = 300
    )
    print(completion)
    response = completion.choices[0].message.content
    
    # Split the response into the description and JSON parts
    description, json_string = response.split('--TDTM--')
    
    # Parse the JSON part of the response
    json_part = '\n'.join(line for line in json_string.split('\n') if line.strip() and not line.strip().startswith('```'))
    outfit_suggestion = json.loads(json_part)
    
    # Construct the final response
    result = {
        "description": description.strip(),
        "outfit_suggestion": outfit_suggestion
    }
    
    return result
