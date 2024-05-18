import os
import json
import pickle
import random
import time
import itertools

import numpy as np
from PIL import Image
import skimage.io as io
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle
from torch.utils.data import Dataset
import webdataset as wds
import csv

from minigpt4.datasets.datasets.base_dataset import BaseDataset
from minigpt4.datasets.datasets.caption_datasets import CaptionDataset


class SlakeDataset(Dataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_path):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.vis_processor = vis_processor['train']
        self.text_processor = text_processor['train']
        self.data = self.create_data(ann_path)

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}",
            "[vqa] Based on the image, respond to this question with a very short and straight answer: {}"
        ]

    def create_data(self, ann_path):
        processed_data = []
  
        with open(ann_path, 'r', newline='') as csv_file:
            data = json.load(csv_file)
            
            for obj in data:
                location = obj['location'].lower().strip()
                if location == 'brain' or location == 'brain_face' or location == 'brain_tissue': # filtering by location
                    image_path = os.path.join(self.vis_root, obj['img_name'])
                    processed_data.append(
                        {'question': str(obj['question']),
                         'answer': str(obj['answer']),
                         'image_path': image_path,
                         'image_id': str(obj['img_id']),
                         'caption': str(obj['caption'])

                         }
                    )
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(sample['image_path']).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(sample["question"])
        answer = self.text_processor(sample["answer"])
        caption = self.text_processor(sample["caption"])

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> Take this caption into account but don't rely on it solely '{}' {}".format(caption, instruction)
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": sample['image_id']
        }
    
