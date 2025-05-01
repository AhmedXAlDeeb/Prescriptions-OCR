import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch import nn
from transformers import TrOCRProcessor, AutoTokenizer, VisionEncoderDecoderModel

from preprocess import HandwrittenExtractor
class HandwrittenTextExtractor:
    def __init__(self, vision_model_path="microsoft/trocr-base-stage1", tokenizer_model="aubmindlab/bert-base-arabertv2", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TrOCRProcessor.from_pretrained(vision_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
        self.processor.tokenizer = self.tokenizer

        self.model = VisionEncoderDecoderModel.from_pretrained(vision_model_path)
        self._configure_model()
        self.model.to(self.device)
        self.model.eval()

    def _configure_model(self):
        self.model.config.decoder_start_token_id = self.tokenizer.cls_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.sep_token_id

        self.model.decoder.config.vocab_size = self.tokenizer.vocab_size
        self.model.config.vocab_size = self.tokenizer.vocab_size
        self.model.decoder.output_projection = nn.Linear(256, self.tokenizer.vocab_size)
        self.model.decoder.model.decoder.embed_tokens = nn.Embedding(self.tokenizer.vocab_size, 256, padding_idx=1)

        self.model.config.max_length = 47
        self.model.config.early_stopping = True
        self.model.config.no_repeat_ngram_size = 3
        self.model.config.length_penalty = 2.0
        self.model.config.num_beams = 8

    # need to be updateed to extract line handwritten text
    def predict_text_from_image(self, image_path):
        cropper = HandwrittenExtractor(image_path)
        cropped_img = cropper.get_cropped()
        if cropped_img is None:
            return ""

        pixel_values = self.processor(cropped_img, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)
            predicted_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return predicted_text

    def batch_predict_from_folder(self, image_folder, batch_size=4):
        results = {}
        images = []
        paths = []

        for file in sorted(os.listdir(image_folder)):
            if not file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
                continue

            full_path = os.path.join(image_folder, file)
            cropper = HandwrittenExtractor(full_path)
            cropped_img = cropper.get_cropped()

            if cropped_img is not None:
                images.append(cropped_img)
                paths.append(full_path)

            if len(images) == batch_size or file == sorted(os.listdir(image_folder))[-1]:
                if images:
                    pixel_values = self.processor(images, return_tensors="pt").pixel_values.to(self.device)
                    with torch.no_grad():
                        generated_ids = self.model.generate(pixel_values)
                        predicted_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
                    for path, text in zip(paths, predicted_texts):
                        results[path] = text
                    images, paths = [], []

        return results
