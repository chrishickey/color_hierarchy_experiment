import os, cv2
import numpy as np
import json
from collections import defaultdict
import torch
import torch.utils.data
from PIL import Image
INT_CLOTHING_CATEGORY = ['bk', 'bag', 'belt', 'boots', 'footwear', 'outer', 'dress', 'sunglasses',
                         'pants', 'top', 'shorts', 'skirt', 'headwear', 'scarf']

class CustomColorAnnotatedDataloader(torch.utils.data.Dataset):
    
    def __init__(self, annotations_dir, image_dir, transforms=None, subcategories=None, color_space='RGB'):
        self.color_space = color_space
        self.annotations_dir = annotations_dir
        self.transforms = transforms
        self.data, categories, colours_dict  = self.get_data(annotations_dir, image_dir, subcategories)
        self.categories = ['bg'] + sorted(list(categories))
        self.colours_dict = colours_dict
        self.colours = sorted(list(colours_dict.keys()))
        self.ordered_files = sorted(list(self.data.keys()))
    
    def get_data(self, annotations_dir, image_dir, subcategories):
        img_data = {}
        categories = set()
        colours = defaultdict(int)
        for file_name in os.listdir(annotations_dir):
            file_name = os.path.join(annotations_dir, file_name)
            with open(file_name, 'r') as fh:
                data = json.load(fh)
            for item in data:
                x1, y1, w, h = item['bbox']
                x2 = w + x1
                y2 = h + y1
                image_id = item['image_id']
                category = item["category_id"]
                colour_name = list(item["colour_description"].keys())[0]
                certainty = round(list(item["colour_description"].values())[0], 3)
                img_path = "{}.jpg".format(os.path.join(image_dir, "{}{}".format("000000", str(image_id))[-7:]))
                if subcategories and category not in subcategories:
                    continue
                if x1 >= x2 or y1 >= y2:
                    #print(img_path)
                    if img_path in img_data:
                        del img_data[img_path]
                    break
                categories.add(category)
                colours[colour_name] += 1
                if img_path in img_data:
                    img_data[img_path]['labels'].append(category)
                    img_data[img_path]['colour_name'].append(colour_name)
                    img_data[img_path]['certainty'].append(certainty)
                    img_data[img_path]['boxes'].append([int(x1), int(y1), int(x2), int(y2)])
                else:
                    img_data[img_path] = {}
                    img_data[img_path]['labels'] = [category]
                    img_data[img_path]['boxes'] = [[int(x1), int(y1), int(x2), int(y2)]]
                    img_data[img_path]['colour_name'] = [colour_name]
                    img_data[img_path]['certainty'] = [certainty]
                    img_data[img_path]['image_id'] = image_id
                    
        return img_data, categories, colours 
            
    def __getitem__(self, idx):
        # load images ad masks
        img_path = self.ordered_files[idx]
        if self.color_space == 'RGB':
            img = Image.open(img_path).convert("RGB")
        elif self.color_space == 'OPP':
            img  = cv2.imread(img_path)
        elif self.color_space == 'YBR':
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2YCrCb)
        elif self.color_space == 'YUV':
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2YUV)

        
        boxes = torch.as_tensor(self.data[img_path]['boxes'], dtype=torch.float32)
        labels = torch.as_tensor([self.categories.index(l) for l in self.data[img_path]['labels']], dtype=torch.int64)
        image_id = torch.tensor(self.data[img_path]['image_id'], dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(self.data[img_path]['boxes']),), dtype=torch.int64)
        certainty = torch.as_tensor(self.data[img_path]['certainty'], dtype=torch.float32)
        colours = torch.as_tensor([self.colours.index(l) for l in self.data[img_path]['colour_name']], dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        target['certainty'] = certainty
        target['colours'] = colours

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.ordered_files)

