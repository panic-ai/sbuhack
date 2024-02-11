from PIL import Image
from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageDraw, ImageFont
import os
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import json
import base64
from openai import OpenAI
client = OpenAI(api_key='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')


cats = ['shirt, blouse', 'top, t-shirt, sweatshirt', 'sweater', 'cardigan', 'jacket', 'vest', 'pants', 'shorts', 'skirt', 'coat', 'dress', 'jumpsuit', 'cape', 'glasses', 'hat', 'headband, head covering, hair accessory', 'tie', 'glove', 'watch', 'belt', 'leg warmer', 'tights, stockings', 'sock', 'shoe', 'bag, wallet', 'scarf', 'umbrella', 'hood', 'collar', 'lapel', 'epaulette', 'sleeve', 'pocket', 'neckline', 'buckle', 'zipper', 'applique', 'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle', 'sequin', 'tassel']


def fix_channels(t):
    """
    Some images may have 4 channels (transparent images) or just 1 channel (black and white images), in order to let the images have only 3 channels. I am going to remove the fourth channel in transparent images and stack the single channel in back and white images.
    :param t: Tensor-like image
    :return: Tensor-like image with three channels
    """
    if len(t.shape) == 2:
        return ToPILImage()(torch.stack([t for i in (0, 0, 0)]))
    if t.shape[0] == 4:
        return ToPILImage()(t[:3])
    if t.shape[0] == 1:
        return ToPILImage()(torch.stack([t[0] for i in (0, 0, 0)]))
    return ToPILImage()(t)

def idx_to_text(i):
    return cats[i]

def get_dominant_color(image, num_clusters=13):
    # Resize image to speed up processing
    small_image = image.resize((50, 50))
    # Convert image data to a sequence of pixels
    np_image = np.array(small_image)
    np_image = np_image.reshape((np_image.shape[0] * np_image.shape[1], 3))
    # Find clusters of colors
    clusters = KMeans(n_clusters=num_clusters).fit(np_image)
    # Count labels to find the most common cluster
    counts = Counter(clusters.labels_)
    # Find the most common cluster center
    center = clusters.cluster_centers_[counts.most_common(1)[0][0]]
    return tuple(center.astype(int))

def rgb_to_color_name(rgb):
    # Define your color mapping
    colors = {
        'red': (255, 0, 0),
        'brown': (165, 42, 42),
        'orange': (255, 165, 0),
        'yellow': (255, 255, 0),
        'pink': (255, 192, 203),
        'purple': (128, 0, 128),
        'violet': (238, 130, 238),
        'indigo': (75, 0, 130),
        'grey': (128, 128, 128),
        'green': (0, 128, 0),
        'blue': (0, 0, 255),
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'turquoise': (64, 224, 208),
        'gold': (255, 215, 0),
        'silver': (192, 192, 192),
        'bronze': (205, 127, 50),
        'beige': (245, 245, 220),
        'teal': (0, 128, 128),
        'maroon': (128, 0, 0),
        'olive': (128, 128, 0),
        'navy': (0, 0, 128),
        'coral': (255, 127, 80),
        'salmon': (250, 128, 114),
        'khaki': (240, 230, 140),
        'lavender': (230, 230, 250),
        'peach': (255, 218, 185),
        'mint': (189, 252, 201),
        'apricot': (251, 206, 177),
        'mustard': (255, 219, 88),
        'chartreuse': (127, 255, 0),
        'taupe': (72, 60, 50),
        'lilac': (200, 162, 200),
        # Add more colors as needed
    }
    color_name = "unknown"
    min_distance = float('inf')
    for name, color_rgb in colors.items():
        distance = sum((s - q) ** 2 for s, q in zip(rgb, color_rgb))  # Euclidean distance
        if distance <= min_distance:
            min_distance = distance
            color_name = name
    return color_name

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def save_segmented_parts(image, outputs, threshold=0.8, output_dir='segmented_parts'):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    to_be_ignored = ['sleeve', 'collar', 'pocket', 'neckline', 'buckle', 'zipper', 'applique',
                     'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle',
                     'sequin', 'tassel']

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    for i, (p, box) in enumerate(zip(probas[keep], bboxes_scaled)):
        category = idx_to_text(p.argmax())
        if category in to_be_ignored:
            continue
        xmin, ymin, xmax, ymax = box.int().tolist()
        cropped_image = image.crop((xmin, ymin, xmax, ymax))

        dominant_color = get_dominant_color(cropped_image)
        color_name = rgb_to_color_name(dominant_color)  # Convert RGB to color name
        print(f"\nCategory: {category}, Color: {color_name}\n")

        print("\nStoreing Segmented Images.......\n\n")
        cropped_image.save(f"{output_dir}/{category}_{color_name}_{i}.jpg")

        return f"{output_dir}/{category}_{color_name}_{i}.jpg"

def text_desc(image_file):
    # Getting the base64 string
    # base64_image =  base64.b64encode(image_file.getvalue()).decode('utf-8')
    base64_image =  base64.b64encode(image_file.read()).decode('utf-8')
    completion = client.chat.completions.create(
    model =  "gpt-4-vision-preview",
    messages= [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Give a breif description of the clothing article in the image. limit to 100 tokens"
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
    # print(completion)
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



MODEL_NAME = "valentinafeve/yolos-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)
image_dir = "userA_clothes/"
from tqdm import tqdm
import os
print()
print()
print()
print('Current Inventory: \n')
for i in os.listdir(image_dir):
    images_path = image_dir + i
    # print(images_path)
    image_name = images_path.split("/")[-1].split(".")[0]
    image = Image.open(open(images_path, "rb"))
    image = fix_channels(ToTensor()(image))
    image = image.resize((600, 800))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
 
    # Call the modified function to save segmented parts
    cropped_image = save_segmented_parts(image, outputs, threshold=0.8, output_dir=f"segmented_parts_{image_dir}")

    # Call the modified function to get the description
    if cropped_image:
        td = text_desc(open(cropped_image, "rb"))
        print(td)
        print()
        print()
    # img = Image.open(images_path + i)
    # img = fix_channels(ToTensor()(img))
    # img = img.resize((600, 800))
    # features = feature_extractor(img.(0))
    # outputs = model(img.unsqueeze(0))
    # for i in range(len(outputs[0]['labels'])):
    #     print(idx_to_text(outputs[0]['labels'][i].item()), outputs[0]['scores'][i].item())
    #     plt.imshow(img)
    #     plt.show()
import os
annotated_dir = 'segmented_parts_userA_clothes'
file_names = []
for file in os.listdir(annotated_dir):
    file_name = file.split(".jpg")[0]
    file_name = file_name.replace(",", " ")
    file_name = file_name.replace("_", " ")

    file_name = ''.join([i for i in file_name if not i.isdigit()])
    file_names.append(file_name)

dress_string = ', '.join(file_names)
# print(concatenated_names)

event_list = [
    'wedding', 'birthday', 'party', 'meeting', 'date', 'casual', 'formal', 'interview', 'work', 'funeral', 'graduation', 'prom', 'concert', 'clubbing', 'beach', 'picnic', 'sport', 'gym', 'travel', 'religious', 'other'
]
from random import random
random_event = event_list[int(random() * len(event_list))]
print()
random_event = input('Enter the event you are attending: ')
# print('radnom event:', random_event)

suggestion = suggest_outfit(random_event, dress_string)
print(suggestion['description'])
for i in suggestion['outfit_suggestion']['outfit']:
    for k, v in i.items():
        print(v,end=' ')
    print()

