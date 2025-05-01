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
        plt.imshow(p["binary_mask"], cmap='blue_image')
        plt.title(f'Mask ')
        plt.axis('off')

        plt.subplot(2, 3, 4)
        plt.imshow(p["blue_emphasized"], cmap='blue_image')
        plt.title('Blue-Emphasized')
        plt.axis('off')

        plt.subplot(2, 3, 5)
        plt.imshow(p["clean_mask"], cmap='blue_image')
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

    def get_blue_cropped_image(self):
        return self._emphasize_blue(self.cropped)

    def get_clean_mask(self):
        return self.mask

    def get_visualized_result(self):
        return self.result_img


class HandwrittenBoxExtractor:
    def __init__(self, image, blue_image):
        self.original_image = image
        self.blue_image = blue_image
        self.binary = None
        self.dilated = None
        self.final_boxes = []
        
    def get_boxes(self):
        return self.final_boxes

    def preprocess(self, visualize):
        # Step 1: Binary inverse thresholding
        _, self.binary = cv2.threshold(self.blue_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.binary = cv2.bitwise_not(self.binary)
        if visualize:
            self._show_image(self.binary, "Binary Thresholded (Otsu Inverse)", cmap='gray')

        # Step 2: Morphological Opening to remove noise
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opened = cv2.morphologyEx(self.binary, cv2.MORPH_OPEN, kernel_open, iterations=1)
        if visualize:
            self._show_image(opened, "After Morphological Opening", cmap='gray')

        # Step 3: Dilation to group characters
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        self.dilated = cv2.dilate(opened, kernel_dilate, iterations=3)
        if visualize:
            self._show_image(self.dilated, "After Dilation", cmap='gray')

    def find_and_filter_boxes(self, visualize):
        contours, _ = cv2.findContours(self.dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) > 300]
        self.final_boxes = self.non_max_suppression_fast(boxes)
        if visualize:
            self.visualize_boxes(self.final_boxes, title="Refined Word Bounding Boxes", color=(0, 255, 0))

    def non_max_suppression_fast(self, boxes, overlapThresh=0.3):
        if len(boxes) == 0:
            return []

        boxes = np.array(boxes)
        pick = []

        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 0] + boxes[:, 2]
        y2 = boxes[:, 1] + boxes[:, 3]

        area = boxes[:, 2] * boxes[:, 3]
        idxs = np.argsort(y2)

        while len(idxs) > 0:
            last = idxs[-1]
            pick.append(last)

            xx1 = np.maximum(x1[last], x1[idxs[:-1]])
            yy1 = np.maximum(y1[last], y1[idxs[:-1]])
            xx2 = np.minimum(x2[last], x2[idxs[:-1]])
            yy2 = np.minimum(y2[last], y2[idxs[:-1]])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)

            overlap = (w * h) / area[idxs[:-1]]
            idxs = np.delete(idxs, np.concatenate(([len(idxs) - 1], np.where(overlap > overlapThresh)[0])))

        return boxes[pick].astype("int")
    def stitch_words_in_line(self, original_image, boxes, padding=10, target_height=None, y_threshold=50):
      """
      Crop words from the image using bounding boxes, group by lines using y-value tolerance, 
      resize to uniform height, and concatenate them horizontally.
      
      Args:
          original_image (np.ndarray): The cropped handwriting image.
          boxes (list): List of (x, y, w, h) tuples for bounding boxes.
          padding (int): Horizontal padding between words.
          target_height (int): Height to resize all words to (optional).
          y_threshold (int): Tolerance to group words on the same line.
          
      Returns:
          np.ndarray: Image with all words stitched in one horizontal line (or stacked by lines if desired).
      """
      if not boxes.any():
          return np.zeros((target_height or 64, 64), dtype=np.uint8)

      # Sort boxes by y
      boxes = sorted(boxes, key=lambda b: b[1])

      # Group boxes into lines
      lines = []
      current_line = [boxes[0]]
      current_y = boxes[0][1]

      for box in boxes[1:]:
          if abs(box[1] - current_y) <= y_threshold:
              current_line.append(box)
          else:
              lines.append(current_line)
              current_line = [box]
              current_y = box[1]
      lines.append(current_line)  # Add the last line

      word_images = []

      for line_boxes in lines:
          # Sort words in the line by x (left to right)
          line_boxes = sorted(line_boxes, key=lambda b: b[0])

          for (x, y, w, h) in line_boxes:
              word_crop = original_image[y:y+h, x:x+w]
              if target_height is not None:
                  scale = target_height / word_crop.shape[0]
                  new_w = int(word_crop.shape[1] * scale)
                  word_crop = cv2.resize(word_crop, (new_w, target_height))
              word_images.append(word_crop)

      # Stitch word images horizontally with padding
      stitched_image = word_images[0]
      for word in word_images[1:]:
          spacer = np.ones((stitched_image.shape[0], padding, 3), dtype=np.uint8) * 255
          stitched_image = np.hstack((stitched_image, spacer, word))

      return stitched_image

    def visualize_boxes(self, boxes, title="Bounding Boxes", color=(0, 255, 0)):
        img_copy = self.original_image.copy()
        for (x, y, w, h) in boxes:
            cv2.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)

        self._show_image(img_copy, title)

    def _show_image(self, img, title, cmap=None):
        plt.figure(figsize=(10, 6))
        if len(img.shape) == 2:
            plt.imshow(img, cmap=cmap if cmap else 'gray')
        else:
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")
        plt.show()

    def run(self, visualize=False):
        self.preprocess(visualize)
        self.find_and_filter_boxes(visualize)

if __name__ == "__main__":
    # path = '/Users/mistaluai/Documents/Github Repos/Prescriptions-OCR/data/test/Beauty_prescription_1.jpg'
    # path = '/Users/mistaluai/Documents/Github Repos/Prescriptions-OCR/data/test/prescription_opth_203.jpg'
    path ='D:\\telemed\Machathon_Prescription_Digtalization_6.00\Machathon_Prescription_Digtalization_6.00\Test_Data_Phase2\Dental_prescription_684.jpg'
    extractor = HandwrittenExtractor(path)
    # extractor.show_debug_plot()

    cropped = extractor.get_cropped_image()
    blue_cropped = extractor.get_blue_cropped_image()

    box_extractor = HandwrittenBoxExtractor(cropped, blue_cropped)
    box_extractor.preprocess(visualize=True)
    box_extractor.find_and_filter_boxes(visualize=False)
    boxes = box_extractor.get_boxes()

    stitched_line_img = box_extractor.stitch_words_in_line(cropped, boxes, target_height=64)
    
    plt.imshow(cv2.cvtColor(stitched_line_img, cv2.COLOR_BGR2RGB))
    plt.figure(figsize=(10, 4))
    plt.imshow(cv2.cvtColor(stitched_line_img, cv2.COLOR_BGR2RGB))
    plt.title("Stitched Line Image")
    plt.axis('off')
    plt.tight_layout()
    plt.show()