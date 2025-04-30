import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import os
import pandas as pd
from tqdm.auto import tqdm
import numpy as np

class OCRDataset(Dataset):
    def __init__(self, image_dir, excel_file, processor):
        self.image_dir = image_dir
        self.processor = processor

        try:
            df = pd.read_excel(excel_file)
            if len(df.columns) < 2:
                raise ValueError("Excel file must have at least 2 columns")

            self.examples = []
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Loading dataset"):
                label = str(row[0])
                image_path = str(row[1]) + ".jpg"

                if os.path.exists(os.path.join(image_dir, image_path)):
                    self.examples.append((image_path, label))
                else:
                    print(f"Warning: Image {image_path} not found")

            if not self.examples:
                raise Exception("No valid examples found in the Excel file")

            print(f"Successfully loaded {len(self.examples)} examples")

        except Exception as e:
            raise Exception(f"Error reading Excel file: {str(e)}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        image_path, text = self.examples[idx]
        try:
            image = Image.open(os.path.join(self.image_dir, image_path)).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            if len(self.examples) > 1:
                return self.__getitem__((idx + 1) % len(self.examples))
            raise

        pixel_values = self.processor(image, return_tensors="pt").pixel_values
        labels = self.processor.tokenizer(text,
                                        padding="max_length",
                                        max_length=128,
                                        truncation=True,
                                        return_tensors="pt").input_ids

        return {
            'pixel_values': pixel_values.squeeze(),
            'labels': labels.squeeze(),
            'text': text
        }