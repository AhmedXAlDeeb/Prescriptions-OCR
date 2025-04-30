import cv2
import numpy as np
from PIL import Image, ImageEnhance

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

    def _emphasize_blue(self, image):
        b, g, r = cv2.split(image)
        return cv2.subtract(b, cv2.addWeighted(r, 0.5, g, 0.5, 0))

    def _crop_handwriting(self, mask):
        y_coords, x_coords = np.where(mask == 255)
        if len(x_coords) > 0:
            x_min, x_max = np.min(x_coords), np.max(x_coords)
            y_min, y_max = np.min(y_coords), np.max(y_coords)
            print(f"[DEBUG] Handwriting bounding box -> x:({x_min}, {x_max}), y:({y_min}, {y_max})")
            x_min = max(0, x_min - self.padding)
            y_min = max(0, y_min - self.padding)
            x_max = min(self.original.shape[1], x_max + self.padding)
            y_max = min(self.original.shape[0], y_max + self.padding)
            self.cropped = self.original[y_min:y_max, x_min:x_max]
            print(f"[DEBUG] Cropped handwriting region shape: {self.cropped.shape}")
            cv2.rectangle(self.result_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    def _generate_mask(self, blue_emphasized):
        """
        Generate a binary mask by applying a moving average (mean blur)
        and thresholding to extract blue handwriting regions.
        """
        # Step 1: Apply moving average (mean blur)
        blurred = cv2.blur(blue_emphasized, (3, 3))  # kernel size can be tuned

        # Step 2: Normalize blurred image
        normalized_blur = cv2.normalize(blurred, None, 0, 255, cv2.NORM_MINMAX)

        # Step 3: Apply threshold to highlight regions with strong blue presence
        _, binary_mask = cv2.threshold(normalized_blur, 150, 255, cv2.THRESH_BINARY)  # threshold value can be tuned
        return binary_mask

    def _morphological_cleaning(self, binary_mask):
        """
        Clean mask using statistical filtering with sliding window:
        removes regions whose local stats deviate from global handwriting patterns.
        """
        h, w = binary_mask.shape
        window_size = 50  # can be tuned
        stride = 10  # controls overlap
        threshold_factor = 3  # how strict the filtering is

        # Global statistics
        global_mean = np.mean(binary_mask)
        global_std = np.std(binary_mask)

        # Output mask initialized to zeros
        cleaned_mask = np.zeros_like(binary_mask)

        for y in range(0, h - window_size + 1, stride):
            for x in range(0, w - window_size + 1, stride):
                window = binary_mask[y:y + window_size, x:x + window_size]
                local_mean = np.mean(window)
                local_std = np.std(window)

                # Heuristic: if local region is "ink-like", copy it
                if (
                        local_mean > global_mean * threshold_factor and
                        local_std > global_std * threshold_factor
                ):
                    cleaned_mask[y:y + window_size, x:x + window_size] = np.maximum(
                        cleaned_mask[y:y + window_size, x:x + window_size],
                        window
                    )

        return cleaned_mask

    def _process(self):
        norm = self._normalize_intensity(self.original)
        blue = self._emphasize_blue(norm)
        # blue_emphasized = cv2.GaussianBlur(blue, (5, 5), 0)
        mask = self._generate_mask(blue)
        clean = self._morphological_cleaning(mask)
        self.mask = clean
        self._crop_handwriting(self.mask)
        self.processed = {
            "normalized": norm,
            "blue_emphasized": blue,
            "binary_mask": mask,
            "clean_mask": clean,
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
        plt.imshow(p["binary_mask"], cmap='gray')
        plt.title(f'Mask ')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(p["blue_emphasized"], cmap='gray')
        plt.title('Blue-Emphasized')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(p["clean_mask"], cmap='gray')
        plt.title(f'Mask ')
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

if __name__ == "__main__":
    # path = '/Users/mistaluai/Documents/Github Repos/Prescriptions-OCR/data/test/Beauty_prescription_1.jpg'
    # path = '/Users/mistaluai/Documents/Github Repos/Prescriptions-OCR/data/test/prescription_opth_203.jpg'
    path = '/Users/mistaluai/Documents/Github Repos/Prescriptions-OCR/data/test/prescription_opth_183.jpg'
    extractor = HandwrittenExtractor(path)
    extractor.show_debug_plot()