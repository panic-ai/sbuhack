from transformers import YolosFeatureExtractor, YolosForObjectDetection
import torch
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image, ImageDraw, ImageFont
import os
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
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

def visualize_predictions(image, outputs, threshold=0.8):
    # keep only predictions with confidence >= threshold
    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold

    # convert predicted boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

    # plot results
    plot_results(image, probas[keep], bboxes_scaled)


# Random colors used for visualization
COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

# for output bounding box post-processing
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

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=c, linewidth=3))
        cl = p.argmax()
        ax.text(xmin, ymin, idx_to_text(cl), fontsize=10,
                bbox=dict(facecolor=c, alpha=0.8))
    plt.axis('off')
    plt.show()


MODEL_NAME = "valentinafeve/yolos-fashionpedia"
feature_extractor = YolosFeatureExtractor.from_pretrained('hustvl/yolos-small')
model = YolosForObjectDetection.from_pretrained(MODEL_NAME)

def get_dominant_color(image, num_clusters=5):
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
        if distance < min_distance:
            min_distance = distance
            color_name = name
    return color_name

def save_segmented_parts(image, outputs, threshold=0.8, output_dir='segmented_parts',save=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    to_be_ignored = ['sleeve', 'collar', 'pocket', 'neckline', 'buckle', 'zipper', 'applique',
                     'bead', 'bow', 'flower', 'fringe', 'ribbon', 'rivet', 'ruffle',
                     'sequin', 'tassel']

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > threshold
    bboxes_scaled = rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)
    filelist = []
    imagelist = []
    categorylist = []
    colorlist = []

    for i, (p, box) in enumerate(zip(probas[keep], bboxes_scaled)):
        category = idx_to_text(p.argmax())
        if category in to_be_ignored:
            continue
        xmin, ymin, xmax, ymax = box.int().tolist()
        cropped_image = image.crop((xmin, ymin, xmax, ymax))

        dominant_color = get_dominant_color(cropped_image)
        color_name = rgb_to_color_name(dominant_color)  # Convert RGB to color name
        
        filelist.append(cropped_image)
        categorylist.append(category)
        imagelist.append(cropped_image)
        colorlist.append(color_name)

        

        if save:
            cropped_image.save(f"{output_dir}/{category}_{color_name}_{i}.jpg")
    return filelist,categorylist,imagelist,colorlist


def complete_process(image):
    image = fix_channels(ToTensor()(image))
    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    return save_segmented_parts(image, outputs, threshold=0.5, output_dir=f"segmented_parts_{directory}")





import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python cloth_detection.py <directory>")
        sys.exit(1)
    
    directory = sys.argv[1]
    output_dir = f"{directory}_classified"
    
    for filename in tqdm(os.listdir(directory)):
        if filename.endswith(".jpg"):
            images_path = os.path.join(directory, filename)
            image_name = images_path.split("/")[-1].split(".")[0]
            image = Image.open(open(images_path, "rb"))
            image = fix_channels(ToTensor()(image))
            inputs = feature_extractor(images=image, return_tensors="pt")
            outputs = model(**inputs)

            save_segmented_parts(image, outputs, threshold=0.5, output_dir=f"segmented_parts_{directory}")
