import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os


class HandwrittenTextProcessor:
    """Processor specifically for cropped handwritten text images using edge detection"""

    def __init__(self, image_path):
        """Initialize with an image path"""
        self.image_path = image_path
        self.original = cv2.imread(image_path)
        if self.original is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        # Storage for processing results
        self.gray = None
        self.filtered = None  # Added to store filtered image
        self.edges = None
        self.dilated_edges = None
        self.character_contours = []
        self.character_boxes = []
        self.word_boxes = []
        self.result_image = None

    def preprocess(self):
        """Convert to grayscale and enhance contrast"""
        # Convert to grayscale
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.gray = clahe.apply(self.gray)

        return self.gray

    def apply_noise_reduction(self, method='gaussian', kernel_size=3, sigma=0):
        """Apply noise reduction filtering to the grayscale image

        Parameters:
        -----------
        method : str
            The filtering method to use: 'gaussian', 'median', 'bilateral', or 'nlmeans'
        kernel_size : int
            Size of the filter kernel
        sigma : float
            Standard deviation for Gaussian or bilateral filter

        Returns:
        --------
        numpy.ndarray
            The filtered image
        """
        if self.gray is None:
            self.preprocess()

        if method == 'gaussian':
            # Gaussian blur - good for general noise reduction
            self.filtered = cv2.GaussianBlur(self.gray, (kernel_size, kernel_size), sigma)

        elif method == 'median':
            # Median filter - excellent for salt and pepper noise
            self.filtered = cv2.medianBlur(self.gray, kernel_size)

        elif method == 'bilateral':
            # Bilateral filter - preserves edges while removing noise
            if sigma == 0:
                sigma = 75  # Default value for bilateral filter
            self.filtered = cv2.bilateralFilter(self.gray, kernel_size, sigma, sigma)

        elif method == 'nlmeans':
            # Non-local means denoising - advanced algorithm for texture preservation
            # Parameters: image, filter strength, template window size, search window size
            h = 10  # Filter strength (higher values remove more noise but might blur details)
            self.filtered = cv2.fastNlMeansDenoising(self.gray, None, h, kernel_size, kernel_size * 2)

        else:
            # Default to Gaussian if method not recognized
            self.filtered = cv2.GaussianBlur(self.gray, (kernel_size, kernel_size), sigma)

        return self.filtered

    def detect_edges(self, low_threshold=50, high_threshold=150):
        """Detect edges using Canny edge detector"""
        # Use filtered image if available, otherwise use gray
        source_image = self.filtered if self.filtered is not None else self.gray

        if source_image is None:
            self.preprocess()
            source_image = self.gray

        # Apply Canny edge detection
        self.edges = cv2.Canny(source_image, low_threshold, high_threshold)

        return self.edges

    def enhance_edges(self, kernel_size=(2, 2), iterations=1):
        """Enhance edges by dilating them slightly"""
        if self.edges is None:
            self.detect_edges()

        # Create kernel for dilation
        kernel = np.ones(kernel_size, np.uint8)

        # Dilate edges to connect nearby components
        self.dilated_edges = cv2.dilate(self.edges, kernel, iterations=iterations)

        return self.dilated_edges

    def find_character_contours(self):
        """Find contours that likely represent characters"""
        if self.dilated_edges is None:
            self.enhance_edges()

        # Find contours in the dilated edge image
        contours, _ = cv2.findContours(self.dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter contours by size to remove noise
        filtered_contours = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)

            # Filter by minimum size (adjust these thresholds based on your images)
            if area > 20 and w > 3 and h > 5:
                filtered_contours.append(contour)
                self.character_boxes.append((x, y, w, h))

        self.character_contours = filtered_contours
        return filtered_contours

    def group_characters_into_words(self, horizontal_gap_factor=0.7, vertical_tolerance_factor=0.4):
        """Group character boxes into word boxes - optimized for Arabic and similar scripts"""
        if not self.character_boxes:
            self.find_character_contours()

        if not self.character_boxes:
            return []

        # Arabic is right-to-left, but we'll sort left-to-right and handle positioning
        sorted_boxes = sorted(self.character_boxes, key=lambda box: box[0])

        # Calculate average character width and height for adaptive spacing
        avg_width = sum(box[2] for box in sorted_boxes) / len(sorted_boxes)
        avg_height = sum(box[3] for box in sorted_boxes) / len(sorted_boxes)

        # For Arabic scripts, we need a much larger horizontal gap allowance
        # horizontal_gap_factor is now treated as absolute pixels if > 5, otherwise as multiplier
        if horizontal_gap_factor > 5:
            horizontal_gap = horizontal_gap_factor  # Use as absolute pixel value
        else:
            horizontal_gap = avg_width * horizontal_gap_factor

        # Vertical tolerance for characters in the same word
        vertical_tolerance = avg_height * vertical_tolerance_factor

        # Use line detection to better group words
        # First, group characters into lines based on vertical position
        y_centers = [(box[1] + box[3] / 2) for box in sorted_boxes]
        # Get more aggressive vertical grouping
        lines = []
        current_line = [0]  # Start with first character index

        # Sort boxes by y-center for line grouping
        indices_by_y = sorted(range(len(y_centers)), key=lambda i: y_centers[i])

        # More aggressive line grouping with larger vertical tolerance
        line_vertical_tolerance = vertical_tolerance * 3
        if indices_by_y:
            current_y = y_centers[indices_by_y[0]]

            for idx in indices_by_y[1:]:
                if abs(y_centers[idx] - current_y) <= line_vertical_tolerance:
                    current_line.append(idx)
                else:
                    lines.append(current_line)
                    current_line = [idx]
                    current_y = y_centers[idx]

            if current_line:
                lines.append(current_line)

        # Process each line to group characters into words
        words = []
        for line in lines:
            # Sort characters in this line by x position
            line_chars = [sorted_boxes[i] for i in line]
            line_chars.sort(key=lambda box: box[0])

            if not line_chars:
                continue

            # Start first word in line
            current_word = [line_chars[0]]
            line_words = []

            # Process remaining characters in line
            for box in line_chars[1:]:
                prev_box = current_word[-1]

                # Calculate horizontal distance
                prev_right = prev_box[0] + prev_box[2]
                current_left = box[0]
                h_distance = current_left - prev_right

                # For Arabic, we use more relaxed horizontal distance
                if h_distance <= horizontal_gap:
                    current_word.append(box)
                else:
                    # Start a new word
                    if current_word:
                        line_words.append(current_word)
                    current_word = [box]

            # Add the last word in the line
            if current_word:
                line_words.append(current_word)

            # Add all words from this line
            words.extend(line_words)

        # Add the last word
        if not words and sorted_boxes:
            words.append([sorted_boxes[0]])

        # Convert groups of character boxes to word boxes
        word_boxes = []
        for word in words:
            # Get the bounding box that contains all characters in the word
            x_min = min(box[0] for box in word)
            y_min = min(box[1] for box in word)
            x_max = max(box[0] + box[2] for box in word)
            y_max = max(box[1] + box[3] for box in word)

            # Create word box
            word_box = (x_min, y_min, x_max - x_min, y_max - y_min)
            word_boxes.append(word_box)

        self.word_boxes = word_boxes
        return word_boxes

    def extract_words(self, padding=5):
        """Extract individual word images from the original with some padding"""
        if not self.word_boxes:
            self.group_characters_into_words()

        word_images = []
        for i, (x, y, w, h) in enumerate(self.word_boxes):
            # Add padding but ensure we don't go outside image boundaries
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(self.original.shape[1], x + w + padding)
            y2 = min(self.original.shape[0], y + h + padding)

            # Extract word image
            word_img = self.original[y1:y2, x1:x2].copy()
            word_images.append((i, word_img))

        return word_images

    def visualize_results(self, show_characters=True, show_words=True):
        """Visualize the detected characters and words"""
        # Create a copy of original for visualization
        result = self.original.copy()

        # Draw character boxes if requested
        if show_characters and self.character_boxes:
            for x, y, w, h in self.character_boxes:
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Draw word boxes if requested
        if show_words and self.word_boxes:
            # Use different colors for each word box to make them easier to distinguish
            colors = [
                (255, 0, 0),  # Red
                (0, 0, 255),  # Blue
                (0, 255, 0),  # Green
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Cyan
                (255, 255, 0),  # Yellow
                (128, 0, 0),  # Maroon
                (0, 0, 128),  # Navy
                (0, 128, 0)  # Dark Green
            ]

            for i, (x, y, w, h) in enumerate(self.word_boxes):
                color = colors[i % len(colors)]
                cv2.rectangle(result, (x, y), (x + w, y + h), color, 2)

        self.result_image = result

        # Create visualization with multiple subplots
        plt.figure(figsize=(15, 15))

        # Original image
        plt.subplot(3, 3, 1)
        plt.imshow(cv2.cvtColor(self.original, cv2.COLOR_BGR2RGB))
        plt.title('Original Image')
        plt.axis('off')

        # Gray image
        plt.subplot(3, 3, 2)
        plt.imshow(self.gray, cmap='gray')
        plt.title('Grayscale')
        plt.axis('off')

        # Filtered image
        plt.subplot(3, 3, 3)
        if self.filtered is not None:
            plt.imshow(self.filtered, cmap='gray')
            plt.title('Filtered (Noise Reduction)')
        else:
            plt.imshow(np.zeros_like(self.gray), cmap='gray')
            plt.title('No Filtering Applied')
        plt.axis('off')

        # Edge detection
        plt.subplot(3, 3, 4)
        plt.imshow(self.edges, cmap='gray')
        plt.title('Edge Detection')
        plt.axis('off')

        # Dilated edges
        plt.subplot(3, 3, 5)
        plt.imshow(self.dilated_edges, cmap='gray')
        plt.title('Dilated Edges')
        plt.axis('off')

        # Result with boxes
        plt.subplot(3, 3, 6)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        if show_characters and show_words:
            title = f'Characters ({len(self.character_boxes)}) and Words ({len(self.word_boxes)})'
        elif show_characters:
            title = f'Characters ({len(self.character_boxes)})'
        elif show_words:
            title = f'Words ({len(self.word_boxes)})'
        else:
            title = 'Result'
        plt.title(title)
        plt.axis('off')

        # Stats and info
        plt.subplot(3, 3, 7)
        info_img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        stats_text = [
            f"Total characters detected: {len(self.character_boxes)}",
            f"Words grouped: {len(self.word_boxes)}",
            f"Avg chars per word: {len(self.character_boxes) / max(1, len(self.word_boxes)):.1f}",
            f"Image dimensions: {self.original.shape[1]}x{self.original.shape[0]}",
        ]

        for i, text in enumerate(stats_text):
            cv2.putText(info_img, text, (20, 40 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)

        plt.imshow(info_img)
        plt.title('Statistics')
        plt.axis('off')

        plt.tight_layout()
        plt.show()

        return result

    def save_word_images(self, output_dir="extracted_words"):
        """Save individual word images to a directory"""
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)

        # Extract word images
        word_images = self.extract_words()

        # Save each word image
        saved_paths = []
        for i, img in word_images:
            # Create filename based on original image name
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            filename = f"{base_name}_word_{i}.jpg"
            output_file = output_path / filename

            # Save the image
            cv2.imwrite(str(output_file), img)
            saved_paths.append(str(output_file))

        print(f"Saved {len(saved_paths)} word images to {output_dir}/")
        return saved_paths

    def process(self, filter_method='bilateral', filter_kernel_size=5, filter_sigma=50,
                edge_low=50, edge_high=150, dilation_kernel=(2, 2),
                h_gap_factor=0.7, v_tolerance_factor=0.4, visualize=True, debug=False):
        """Run the complete processing pipeline with noise reduction"""
        # Step 1: Preprocess the image
        self.preprocess()

        # Step 2: Apply noise reduction
        self.apply_noise_reduction(method=filter_method,
                                   kernel_size=filter_kernel_size,
                                   sigma=filter_sigma)

        # Step 3: Detect edges
        self.detect_edges(low_threshold=edge_low, high_threshold=edge_high)

        # Step 4: Enhance edges
        self.enhance_edges(kernel_size=dilation_kernel)

        # Step 5: Find character contours
        self.find_character_contours()

        # Print debug info if requested
        if debug:
            print(f"Found {len(self.character_boxes)} character boxes")
            print(
                f"Average character width: {sum(box[2] for box in self.character_boxes) / max(1, len(self.character_boxes)):.1f} pixels")
            print(f"Using horizontal gap factor: {h_gap_factor}")

            if h_gap_factor > 5:
                print(f"Using absolute pixel gap: {h_gap_factor} pixels")
            else:
                avg_width = sum(box[2] for box in self.character_boxes) / max(1, len(self.character_boxes))
                print(f"Using relative gap: {avg_width * h_gap_factor:.1f} pixels")

        # Step 6: Group characters into words
        self.group_characters_into_words(
            horizontal_gap_factor=h_gap_factor,
            vertical_tolerance_factor=v_tolerance_factor
        )

        # Step 7: Visualize results if requested
        if visualize:
            self.visualize_results()

        return {
            'character_boxes': self.character_boxes,
            'word_boxes': self.word_boxes,
            'result_image': self.result_image
        }


# Example usage
if __name__ == "__main__":
    # Path to your cropped handwritten text image
    image_path = "crop_Dental_prescription_516.jpg"

    # Process the image
    processor = HandwrittenTextProcessor(image_path)

    # Run with default parameters and noise reduction
    results = processor.process(
        filter_method='bilateral',  # 'gaussian', 'median', 'bilateral', or 'nlmeans'
        filter_kernel_size=9,  # Size of the filter kernel
        filter_sigma=80,  # Parameter for bilateral filter
        edge_low=40,  # Lower threshold for edge detection
        edge_high=160,  # Upper threshold for edge detection
        dilation_kernel=(4, 4),  # Good balance for both scripts
        h_gap_factor=80,  # Absolute pixel distance
        v_tolerance_factor=0.5,  # Vertical alignment tolerance
        debug=True  # Print debugging information
    )

    # Save individual word images
    processor.save_word_images()

    # Display a summary
    print(f"Detected {len(processor.character_boxes)} characters")
    print(f"Grouped into {len(processor.word_boxes)} words")