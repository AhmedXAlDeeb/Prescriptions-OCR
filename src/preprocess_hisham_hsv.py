import cv2
import numpy as np
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt


class PrescriptionML:
    """Machine learning enhancements for prescription processing"""

    def __init__(self):
        # Pre-trained model paths would go here
        self.model_loaded = False
        try:
            # You would load models here in practice
            # self.segmentation_model = load_model('path_to_model')
            self.model_loaded = False  # Set to True when models are actually loaded
        except:
            print("ML models not available. Falling back to traditional CV methods.")

    def segment_prescription_regions(self, image):
        """Use semantic segmentation to identify prescription regions"""
        if not self.model_loaded:
            # Fallback to traditional method
            return None

        # This would be implemented with your trained model
        # preprocessed = preprocess_for_model(image)
        # segmentation_mask = self.segmentation_model.predict(preprocessed)
        # return post_process_segmentation(segmentation_mask)

        # For now, just return a placeholder
        return np.zeros_like(image[:, :, 0])

    def enhance_image_quality(self, image):
        """Use a denoising autoencoder or similar to enhance image quality"""
        # In a real implementation, you'd use a pre-trained model
        # Enhanced code would go here

        # For now, just apply traditional denoising
        return cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)


class EnhancedPrescriptionProcessor:
    def __init__(self, image_path):
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # Initialize components
        self.ml_helper = PrescriptionML()

        # Results storage
        self.processed_image = None
        self.blue_emphasized = None
        self.text_mask = None
        self.text_boxes = []
        self.text_lines = []
        self.stitched_result = None

    def find_text_regions_with_cc(self, binary_image):
        """Use connected components analysis to find text regions"""
        # Apply connected components with stats
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_image, connectivity=8
        )

        # Filter components by size to remove noise
        min_area = 50  # Minimum pixel area to be considered text
        max_area = binary_image.shape[0] * binary_image.shape[1] // 4  # Max 1/4 of image

        boxes = []
        for i in range(1, num_labels):  # Skip label 0 (background)
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            w = stats[i, cv2.CC_STAT_WIDTH]
            h = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]

            if min_area < area < max_area:
                # Filter by aspect ratio to avoid extreme rectangles
                aspect_ratio = w / h if h > 0 else 0
                if 0.1 < aspect_ratio < 10:  # Reasonable aspect ratio for text
                    boxes.append((x, y, w, h))

        return boxes

    def detect_text_lines(self, boxes, image_height, y_tolerance_factor=0.15):
        """
        More sophisticated text line detection that adapts to text size

        Args:
            boxes: List of (x, y, w, h) tuples representing word bounding boxes
            image_height: Height of the original image
            y_tolerance_factor: Fraction of text height to use as tolerance

        Returns:
            List of lists, where each inner list contains boxes for one text line
        """
        if not boxes:
            return []

        # Sort boxes by y-coordinate (top to bottom)
        boxes = sorted(boxes, key=lambda box: box[1])

        lines = []
        current_line = [boxes[0]]

        # Calculate average text height for adaptive tolerance
        text_heights = [h for _, _, _, h in boxes]
        avg_height = sum(text_heights) / len(text_heights)

        # Adaptive y-tolerance based on text size
        y_tolerance = max(int(avg_height * y_tolerance_factor), 10)

        last_y_bottom = boxes[0][1] + boxes[0][3]  # y + height

        for box in boxes[1:]:
            current_y = box[1]

            # If this box starts below the bottom of the last line (with tolerance)
            if current_y > last_y_bottom + y_tolerance:
                # Start a new line
                lines.append(current_line)
                current_line = [box]
                last_y_bottom = box[1] + box[3]
            else:
                # Add to current line and maybe extend the line's bottom coordinate
                current_line.append(box)
                last_y_bottom = max(last_y_bottom, box[1] + box[3])

        # Add the last line
        if current_line:
            lines.append(current_line)

        # Sort boxes within each line by x-coordinate (left to right)
        for i in range(len(lines)):
            lines[i] = sorted(lines[i], key=lambda box: box[0])

        return lines

    def detect_text_with_mser(self, gray_image):
        """Detect text regions using MSER algorithm"""
        # Create MSER detector
        mser = cv2.MSER_create(
            delta=5,  # Stability threshold
            min_area=60,  # Min area of detected region
            max_area=14400,  # Max area of detected region
            max_variation=0.25  # Maximum variation
        )

        # Detect regions
        regions, _ = mser.detectRegions(gray_image)

        # Create mask and draw regions
        mask = np.zeros_like(gray_image)
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            cv2.drawContours(mask, [hull], 0, 255, -1)

        # Apply morphological operations to connect nearby text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))  # Horizontal kernel
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours on the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding boxes
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if w > 20 and h > 8:  # Filter small noise
                boxes.append((x, y, w, h))

        return boxes, mask

    def preprocess(self):
        """Enhanced preprocessing pipeline"""
        # Step 1: Basic image enhancement
        denoised = self.ml_helper.enhance_image_quality(self.original)

        # Step 2: Convert to proper color spaces
        rgb_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB)
        lab_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2LAB)

        # Step 3: Isolate blue components using multiple color spaces
        # HSV blue extraction
        hsv_image = cv2.cvtColor(denoised, cv2.COLOR_BGR2HSV)
        lower_blue_hsv = np.array([90, 50, 50])
        upper_blue_hsv = np.array([130, 255, 255])
        blue_mask_hsv = cv2.inRange(hsv_image, lower_blue_hsv, upper_blue_hsv)

        # RGB blue channel emphasis
        b, g, r = cv2.split(denoised)
        blue_emphasis_rgb = cv2.subtract(b, cv2.addWeighted(r, 0.5, g, 0.5, 0))

        # LAB color space - 'b' channel separates blue-yellow
        l, a, b_channel = cv2.split(lab_image)
        _, blue_mask_lab = cv2.threshold(a, 127, 255, cv2.THRESH_BINARY_INV)

        # Step 4: Combine blue detection results
        combined_blue = cv2.bitwise_or(
            blue_mask_hsv,
            cv2.bitwise_or(blue_emphasis_rgb, blue_mask_lab)
        )

        # Step 5: Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_blue = clahe.apply(combined_blue)

        # Final cleanup with morphological operations
        kernel = np.ones((2, 2), np.uint8)
        clean_blue = cv2.morphologyEx(enhanced_blue, cv2.MORPH_OPEN, kernel)

        self.blue_emphasized = clean_blue
        self.processed_image = denoised
        return clean_blue

    def detect_text_regions(self):
        """Detect text regions using multiple methods and combine results"""
        # Method 1: Connected Components
        cc_boxes = self.find_text_regions_with_cc(self.blue_emphasized)

        # Method 2: MSER detection
        mser_boxes, mser_mask = self.detect_text_with_mser(self.blue_emphasized)

        # Method 3: Traditional contour detection
        _, binary = cv2.threshold(self.blue_emphasized, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contour_boxes = self.detect_contour_boxes(binary)

        # Combine and filter boxes
        all_boxes = cc_boxes + mser_boxes + contour_boxes
        filtered_boxes = self.filter_overlapping_boxes(all_boxes)

        self.text_boxes = filtered_boxes
        return filtered_boxes

    def detect_contour_boxes(self, binary_image):
        """Traditional contour-based text detection"""
        # Dilate to connect nearby text components
        kernel = np.ones((5, 2), np.uint8)
        dilated = cv2.dilate(binary_image, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size and shape
        boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter by size
            if area > 100 and w > 10 and h > 5:
                boxes.append((x, y, w, h))

        return boxes

    def filter_overlapping_boxes(self, boxes, overlap_threshold=0.7):
        """Remove highly overlapping boxes using Non-Maximum Suppression"""
        if not boxes:
            return []

        # Convert to the format expected by NMS
        box_array = np.array(boxes)

        # Calculate box areas
        areas = box_array[:, 2] * box_array[:, 3]

        # Sort boxes by y-coordinate (top to bottom)
        indices = np.argsort(box_array[:, 1])

        keep = []
        while len(indices) > 0:
            # Pick the box with smallest y-coordinate
            current = indices[0]
            keep.append(current)

            # Calculate IoU with remaining boxes
            overlap_indices = []
            for i in range(1, len(indices)):
                idx = indices[i]

                # Calculate intersection coordinates
                x1 = max(box_array[current, 0], box_array[idx, 0])
                y1 = max(box_array[current, 1], box_array[idx, 1])
                x2 = min(box_array[current, 0] + box_array[current, 2],
                         box_array[idx, 0] + box_array[idx, 2])
                y2 = min(box_array[current, 1] + box_array[current, 3],
                         box_array[idx, 1] + box_array[idx, 3])

                # Calculate intersection area
                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                intersection = w * h

                # Calculate IoU
                union = areas[current] + areas[idx] - intersection
                iou = intersection / union if union > 0 else 0

                # If overlap is high, mark for removal
                if iou > overlap_threshold:
                    overlap_indices.append(i)

            # Remove overlapping boxes
            indices = np.delete(indices, [0] + overlap_indices)

        return box_array[keep].tolist()

    def group_text_into_lines(self):
        """Group detected text boxes into lines"""
        self.text_lines = self.detect_text_lines(
            self.text_boxes,
            self.original.shape[0]
        )
        return self.text_lines

    def extract_and_stitch_text(self, target_height=64, horizontal_gap=10):
        """Extract text regions and stitch them into readable format"""
        if not self.text_lines:
            self.group_text_into_lines()

        # Process each line
        line_images = []
        for line in self.text_lines:
            # Sort boxes in line from left to right
            line = sorted(line, key=lambda box: box[0])

            # Extract and resize each text region
            word_images = []
            for x, y, w, h in line:
                word_img = self.original[y:y + h, x:x + w]

                # Skip empty regions
                if word_img.size == 0:
                    continue

                # Resize to target height
                scale = target_height / h if h > 0 else 1
                new_w = max(1, int(w * scale))
                resized = cv2.resize(word_img, (new_w, target_height))

                word_images.append(resized)

            # Stitch words in this line with gaps
            if word_images:
                line_image = word_images[0]
                for word in word_images[1:]:
                    # Create gap
                    gap = np.ones((target_height, horizontal_gap, 3), dtype=np.uint8) * 255
                    # Append word
                    line_image = np.hstack((line_image, gap, word))

                line_images.append(line_image)

        # Stack lines vertically with spacing
        if line_images:
            # Find maximum width among all line images
            max_width = max(line.shape[1] for line in line_images)

            # First, ensure all lines have the same width
            normalized_lines = []
            for line in line_images:
                if line.shape[1] < max_width:
                    # Pad shorter image to match max_width
                    padding = np.ones((line.shape[0], max_width - line.shape[1], 3), dtype=np.uint8) * 255
                    line = np.hstack((line, padding))
                normalized_lines.append(line)

            # Create vertical gap with correct width
            vertical_gap = np.ones((20, max_width, 3), dtype=np.uint8) * 255

            # Start with first line
            result = normalized_lines[0]

            # Add remaining lines with vertical gaps
            for line in normalized_lines[1:]:
                result = np.vstack((result, vertical_gap, line))

            self.stitched_result = result
            return result

        return np.zeros((target_height, 100, 3), dtype=np.uint8)

    def visualize_results(self):
        """Generate visualization of all processing steps"""
        # Create figure with subplots
        plt.figure(figsize=(20, 15))

        # Original image
        plt.subplot(3, 2, 1)
        plt.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        plt.title('Original Prescription')
        plt.axis('off')

        # Blue emphasized
        plt.subplot(3, 2, 2)
        plt.imshow(self.blue_emphasized, cmap='gray')
        plt.title('Blue Handwriting Emphasized')
        plt.axis('off')

        # Detected text boxes
        plt.subplot(3, 2, 3)
        box_vis = self.original.copy()
        for x, y, w, h in self.text_boxes:
            cv2.rectangle(box_vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
        plt.imshow(cv2.cvtColor(box_vis, cv2.COLOR_BGR2RGB))
        plt.title(f'Detected Text Regions ({len(self.text_boxes)} boxes)')
        plt.axis('off')

        # Text lines visualization
        plt.subplot(3, 2, 4)
        line_vis = self.original.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255),
                  (255, 255, 0), (255, 0, 255), (0, 255, 255)]

        for i, line in enumerate(self.text_lines):
            color = colors[i % len(colors)]
            for x, y, w, h in line:
                cv2.rectangle(line_vis, (x, y), (x + w, y + h), color, 2)

        plt.imshow(cv2.cvtColor(line_vis, cv2.COLOR_BGR2RGB))
        plt.title(f'Text Lines Grouping ({len(self.text_lines)} lines)')
        plt.axis('off')

        # Final stitched result
        plt.subplot(3, 2, 5)
        if self.stitched_result is not None:
            plt.imshow(cv2.cvtColor(self.stitched_result, cv2.COLOR_BGR2RGB))
            plt.title('Stitched Text Result')
        else:
            plt.title('No Stitched Result Available')
        plt.axis('off')

        # Text extraction visualization
        plt.subplot(3, 2, 6)
        # Create an information panel
        info_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        stats_text = [
            f"Total text boxes: {len(self.text_boxes)}",
            f"Number of text lines: {len(self.text_lines)}",
            f"Image dimensions: {self.original.shape[1]}x{self.original.shape[0]}",
            "Processing complete!"
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(info_img, text, (20, 40 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        plt.imshow(info_img)
        plt.title('Extraction Statistics')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

    def process(self, visualize=True):
        """Run full processing pipeline"""
        # Step 1: Preprocess image
        self.preprocess()

        # Step 2: Detect text regions
        self.detect_text_regions()

        # Step 3: Group into lines
        self.group_text_into_lines()

        # Step 4: Extract and stitch text
        self.extract_and_stitch_text()

        # Step 5: Visualize results if requested
        if visualize:
            self.visualize_results()

        return {
            'processed_image': self.processed_image,
            'blue_emphasized': self.blue_emphasized,
            'text_boxes': self.text_boxes,
            'text_lines': self.text_lines,
            'stitched_result': self.stitched_result
        }


if __name__ == "__main__":
    # Path to your prescription image
    image_path = "Beauty_prescription_100.jpg"

    # Process the image
    processor = EnhancedPrescriptionProcessor(image_path)
    results = processor.process(visualize=True)

    # Save the stitched result
    if results['stitched_result'] is not None:
        cv2.imwrite("extracted_handwriting.jpg", results['stitched_result'])
        print(f"Extracted handwriting saved to extracted_handwriting.jpg")

    # You could now pass the extracted handwriting to an OCR system
    # For example:
    # import pytesseract
    # text = pytesseract.image_to_string(results['stitched_result'])
    # print("Extracted text:", text)