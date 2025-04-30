import os
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class HandwrittenTextExtractor:
    def __init__(self, model_path, device=None):
        """
        Initialize the extractor with TrOCR model and processor
        """
        self.processor = TrOCRProcessor.from_pretrained(model_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.model.eval()

    def predict_text_from_image(self, image_path):
        """
        Predict text from a single image file
        """
        cropped_img = self.preprocess_and_crop_handwriting(image_path)
        if cropped_img is None:
            return ""
        pixel_values = self.processor(cropped_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return predicted_text

    def batch_predict_from_folder(self, image_folder, batch_size=4):
        """
        Predict text from all images in a folder
        """
        results = {}
        images = []
        image_paths = []
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_files = [f for f in os.listdir(image_folder)
                       if f.lower().endswith(valid_extensions)]

        for i, file_name in enumerate(image_files):
            path = os.path.join(image_folder, file_name)
            cropped = self.preprocess_and_crop_handwriting(path)
            if cropped is not None:
                images.append(cropped)
                image_paths.append(path)

            # Process in batches
            if len(images) == batch_size or i == len(image_files) - 1:
                if images:
                    pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(self.device)
                    with torch.no_grad():
                        generated_ids = self.model.generate(pixel_values)
                        predicted_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    for img_path, text in zip(image_paths, predicted_texts):
                        results[img_path] = text
                    images = []
                    image_paths = []

        return results
