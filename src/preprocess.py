import cv2
import numpy as np
from PIL import Image, ImageEnhance
from torchvision import transforms
import os

import matplotlib.pyplot as plt
class HandwrittenPreprocessor:
    def __init__(self, target_size=(384, 384)):
        self.target_size = target_size

    def enhance_contrast(self, image):
        """Enhance image contrast"""
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(1.5)

    def denoise(self, image):
        """Remove noise from image"""
        img_array = np.array(image)
        denoised = cv2.fastNlMeansDenoisingColored(img_array, None, 10, 10, 7, 21)
        return Image.fromarray(denoised)

    def remove_background(self, image):
        """Remove background using adaptive thresholding"""
        img_array = np.array(image)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

        binary = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        kernel = np.ones((2,2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(result)

class HandwrittenExtractor:
    def __init__(self, image_path, padding=100):
        self.image_path = image_path
        self.padding = padding
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")
        self.processed = None
        self.mask = None
        self.cropped = None
        self.result_img = self.original.copy()
        self._process()

    def _normalize_intensity(self, image):
        return cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    def _apply_clahe(self, image):
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        lab_eq = cv2.merge((l_eq, a, b))
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

    def _emphasize_blue(self, image):
        b, g, r = cv2.split(image)
        return cv2.subtract(b, cv2.addWeighted(r, 0.5, g, 0.5, 0))

    def _adaptive_threshold(self, image):
        max_val = np.max(image)
        lower_thresh = 0.35 * max_val
        upper_thresh = 0.75 * max_val
        _, weak = cv2.threshold(image, lower_thresh, 255, cv2.THRESH_BINARY)
        _, strong = cv2.threshold(image, upper_thresh, 255, cv2.THRESH_BINARY)
        binary = cv2.bitwise_or(strong, cv2.bitwise_and(weak, cv2.dilate(strong, None)))
        return binary, lower_thresh, upper_thresh

    def _morphological_cleaning(self, binary_mask):
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        return cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)

    def _crop_handwriting(self, mask):
        y_coords, x_coords = np.where(mask == 255)
        if len(x_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            x_min = max(0, x_min - self.padding)
            y_min = max(0, y_min - self.padding)
            x_max = min(self.original.shape[1], x_max + self.padding)
            y_max = min(self.original.shape[0], y_max + self.padding)
            self.cropped = self.original[y_min:y_max, x_min:x_max]
            cv2.rectangle(self.result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    def _process(self):
        norm = self._normalize_intensity(self.original)
        eq = self._apply_clahe(norm)
        blue = self._emphasize_blue(eq)
        blue = cv2.GaussianBlur(blue, (3, 3), 0)
        mask, low_t, high_t = self._adaptive_threshold(blue)
        clean = self._morphological_cleaning(mask)
        self.mask = clean
        self._crop_handwriting(clean)
        self.processed = {
            "normalized": norm,
            "equalized": eq,
            "blue_emphasized": blue,
            "binary_mask": mask,
            "clean_mask": clean,
            "lower_thresh": low_t,
            "upper_thresh": high_t
        }

    def show_debug_plot(self):
        p = self.processed
        plt.figure(figsize=(18, 12))

        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(2, 3, 2)
        plt.imshow(cv2.cvtColor(p["normalized"], cv2.COLOR_BGR2RGB))
        plt.title('Normalized Image')
        plt.axis('off')

        plt.subplot(2, 3, 3)
        plt.imshow(cv2.cvtColor(p["equalized"], cv2.COLOR_BGR2RGB))
        plt.title('Contrast Equalized')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(p["blue_emphasized"], cmap='gray')
        plt.title('Blue-Emphasized')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(p["clean_mask"], cmap='gray')
        plt.title(f'Mask (thresh = {p["lower_thresh"]:.0f}/{p["upper_thresh"]:.0f})')
        plt.axis('off')

        plt.subplot(2, 3, 6)
        if self.cropped is not None:
            plt.imshow(cv2.cvtColor(self.cropped, cv2.COLOR_BGR2RGB))
        else:
            plt.imshow(np.zeros((10, 10, 3)))
        plt.title('Cropped Handwriting')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def get_cropped_image(self):
        return self.cropped

    def get_clean_mask(self):
        return self.mask

    def get_visualized_result(self):
        return self.result_img
