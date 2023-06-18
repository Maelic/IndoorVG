import json
from visual_genome import api as vg
import pandas as pd

with open("original_annotations/objects.json") as file:    # objects.json locally
    obj_data=json.load(file)
    
with open("original_annotations/region_descriptions.json") as file:    # objects.json locally
    region_data=json.load(file)

regions = []
images = []
for i in range(len(obj_data)):
    if i%1000==0:
        print("processed images: ", i, "out of ", len(obj_data))
    region_sentences = []      # descriptions for areas of the chosen image
    image_id=obj_data[i]['image_id']
    try:
        region = region_data[i]['regions']
        # sanity check
        assert region_data[i]['id'] == image_id
        for j in region:
            region_sentences.append(j['phrase'].lower())
        if region_sentences:
            images.append(image_id)
            regions.append(region_sentences) 
    except IndexError:
        continue
        
images_regions = pd.DataFrame(list(zip(images, regions)), columns =['Image_id', 'region_sentences'])
images_regions.to_csv('image_regions.csv', index=False)