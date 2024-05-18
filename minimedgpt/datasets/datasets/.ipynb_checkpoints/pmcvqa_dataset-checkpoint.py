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


class PMCVQADataset(Dataset):
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
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

    def create_data(self, ann_path):
        processed_data = []
  
        with open(ann_path, 'r', newline='') as csv_file:
            data = csv.DictReader(csv_file)
            
            for obj in data:
                image_path = os.path.join(self.vis_root, obj['Figure_path'])
                processed_data.append(
                    {'question': str(obj['Question']),
                     'answer': str(obj['Answer']),
                     'image_path': image_path,
                     'image_id': str(obj['Figure_path']),
                     }
                )
                # q = f"{str(obj['Question']).strip()} {str(obj['Choice A']).strip()} {str(obj['Choice B']).strip()} {str(obj['Choice C']).strip()} {str(obj['Choice D']).strip()}"
                # processed_data.append(
                #     {'question': q,
                #      'answer': str(obj['Answer']),
                #      'image_path': image_path,
                #      'image_id': str(obj['Figure_path']),
                #      }
                # )
                # processed_data.append(
                #     {'question': q,
                #      'answer': str(obj['Answer_label']),
                #      'image_path': image_path,
                #      'image_id': str(obj['Figure_path']),
                #      }
                # )
                # processed_data.append(
                #     {'question': q,
                #      'answer': str(obj['Answer_label']) + ': ' + str(obj['Answer']),
                #      'image_path': image_path,
                #      'image_id': str(obj['Figure_path']),
                #      }
                # )
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sample = self.data[index]
        image = Image.open(sample['image_path']).convert("RGB")
        image = self.vis_processor(image)
        question = self.text_processor(sample["question"])
        answer = self.text_processor(sample["answer"])

        instruction = random.choice(self.instruction_pool).format(question)
        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        return {
            "image": image,
            "instruction_input": instruction,
            "answer": answer,
            "image_id": sample['image_id']
        }
    
