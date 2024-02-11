import base64

from openai import OpenAI

client = OpenAI(api_key='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')

# Function to encode the image
def text_desc(image_file):
    # Getting the base64 string
    base64_image =  base64.b64encode(image_file).decode('utf-8')
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
