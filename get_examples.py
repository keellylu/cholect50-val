from overeasy import *
from PIL import Image
from pydantic import BaseModel, Field
from enum import Enum
import json
import os
import random

class ToolType(Enum):
    GRASPER = "grasper"
    BIPOLAR = "bipolar"
    HOOK = "hook"
    SCISSORS = "scissors"
    CLIPPER = "clipper"
    IRRIGATOR = "irrigator"

def get_tool_name(tool):
    return tool.value

class Tool(BaseModel):
    tool: ToolType = Field(description="Correctly assign one of the predefined surgical tool types to the tool.")

results = {
    "grasper": [], 
    "bipolar": [], 
    "hook": [], 
    "scissors": [], 
    "clipper": [], 
    "irrigator": []
}

# Retrieve id mappings
label_mappings_path = "/home/azureuser/customers/sdsc/overeasy-internal/cholect50-challenge-val/labels/VID74.json"
with open(label_mappings_path, 'r') as file:
    label_mappings = json.load(file)

instrument_mappings = {v: k for k, v in label_mappings["categories"]["instrument"].items()}
triplet_mappings = {v: k for k, v in label_mappings["categories"]["triplet"].items()}

vid74_path = "/home/azureuser/customers/sdsc/overeasy-internal/cholect50-challenge-val/labels/VID74.json"
video_path = "/home/azureuser/customers/sdsc/overeasy-internal/cholect50-challenge-val/videos/VID74"

with open(vid74_path, 'r') as file:
    vid74_mappings = json.load(file)

v74_triplet_mappings = {v: k for k, v in vid74_mappings["categories"]["triplet"].items()}
v74_frame_mappings = vid74_mappings["annotations"]

frame_index = 0
for filename in os.listdir(video_path):
    frame = filename[:-4].lstrip('0')

    if filename.endswith(".png"):
        bb_detections = []
        correct_detections = []
        
        if frame in v74_frame_mappings:
            correct_detections = v74_frame_mappings[frame]
        
        for detection in correct_detections:
            triplet_index = detection[0]
            x1, y1, x2, y2 = detection[3], detection[4], detection[5], detection[6]
            bb_detections = [x1, y1, x2, y2]
            triplet_name = label_mappings["categories"]["triplet"][str(triplet_index)]
            tool, verb, target = triplet_name.split(",")
            results[tool].append({"filename": filename, "bb": bb_detections})

grasper_images = results["grasper"]
bipolar_images = results["bipolar"]
hook_images = results["hook"]
scissors_images = results["scissors"]
clipper_images = results["clipper"]
irrigator_images = results["irrigator"]

def get_crop(image_name, bb, tool_name):
    x, y, width, height = bb
    video_path = "/home/azureuser/customers/sdsc/overeasy-internal/cholect50-challenge-val/videos/VID74"
    image_path = video_path + "/" + image_name
    
    with Image.open(image_path) as im:
        im_width, im_height = im.size
        x1, y1, x2, y2 = x, y, x + width, y + height
        im_crop = im.crop((x1 * im_width, y1 * im_height, x2 * im_width, y2 * im_height))
        im_crop.save("example_photos/all/" + tool_name + ".png")

tool_images = {
    "grasper": grasper_images,
    "hook": hook_images,
    "clipper": clipper_images,
    "scissors": scissors_images,
    "irrigator": irrigator_images,
    "bipolar": bipolar_images,
}

tool_counts = {
    "grasper": 0,
    "hook": 0,
    "clipper": 0,
    "scissors": 0,
    "irrigator": 0,
    "bipolar": 0,
}

for tool_name, images in tool_images.items():
    for i in range(1, 6):
        try:
            selected_image = random.choice(images)
            get_crop(selected_image["filename"], selected_image["bb"], f"{tool_name}{tool_counts[tool_name]}")
            tool_counts[tool_name] += 1
        except:
            continue