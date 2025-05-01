import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
from tqdm import tqdm

class HandwrittenOCRDataset(Dataset):
    def __init__(self, image_dir, excel_file, processor, target_height=64):
        """
        Args:
            image_dir (str): Path to image directory.
            excel_file (str): Excel file with two columns [label, image_id].
            processor (transformer processor): Huggingface processor for image & text.
            target_height (int): Desired height of stitched word line.
        """
        self.image_dir = image_dir
        self.processor = processor
        self.target_height = target_height

        try:
            df = pd.read_excel(excel_file)
            if len(df.columns) < 2:
                raise ValueError("Excel file must have at least 2 columns [label, image_id]")

            self.examples = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
                label = str(row[0])
                image_filename = str(row[1]) + ".jpg"
                full_path = os.path.join(image_dir, image_filename)

                if os.path.exists(full_path):
                    self.examples.append((image_filename, label))
                else:
                    print(f"Warning: Image {image_filename} not found")

            if not self.examples:
                raise Exception("No valid examples found in the Excel file")

            print(f"Successfully loaded {len(self.examples)} examples")

        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_filename, text = self.examples[idx]
        image_path = os.path.join(self.image_dir, image_filename)

        try:
            bgr_image = cv2.imread(image_path)
            if bgr_image is None:
                raise ValueError("cv2.imread failed")

            # === Custom Preprocessing ===
            from preprocess import HandwrittenExtractor, HandwrittenBoxExtractor  # update as needed

            extractor = HandwrittenExtractor(bgr_image)
            cropped = extractor.get_cropped_image()
            blue_cropped = extractor.get_blue_cropped_image()

            box_extractor = HandwrittenBoxExtractor(cropped, blue_cropped)
            box_extractor.preprocess(visualize=False)
            box_extractor.find_and_filter_boxes(visualize=False)
            boxes = box_extractor.get_boxes()
            stitched_img = box_extractor.stitch_words_in_line(cropped, boxes, target_height=self.target_height)

            # === Convert to RGB PIL for processor ===
            rgb_img = cv2.cvtColor(stitched_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

        except Exception as e:
            print(f"Error processing image {image_filename}: {str(e)}")
            if len(self.examples) > 1:
                return self.__getitem__((idx + 1) % len(self.examples))
            raise

        # === Huggingface processor ===
        pixel_values = self.processor(images=pil_img, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=128,
            truncation=True,
            return_tensors="pt"
        ).input_ids

        return {
            'pixel_values': pixel_values.squeeze(),
            'labels': labels.squeeze(),
            'text': text
        }
