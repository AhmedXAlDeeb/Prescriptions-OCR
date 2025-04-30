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


def directionalHistogram(img, direction='H'):
  # a function which outputs the intensity histogram for a given image along 
  #x or y directions

    (w,h) = img.shape
    sum = []
    pixel_count=0

    if(direction=='H'):
        for j in range(w-1):
          for i in range(h-1):
            pixel=img[j,i]
            if(pixel==255):
              pixel_count+=1
          sum.append(pixel_count)
          pixel_count=0

    else:
       for j in range(h-1):
          for i in range(w-1):
            pixel=img[i,j]
            if(pixel==255):
              pixel_count+=1
          sum.append(pixel_count)
          pixel_count=0

    return sum

##############################################################

def smoothHist(hist,kernel_size):
  # A function to smooth out the noise in intensity histograms of an image
  kernel = np.ones(kernel_size) / kernel_size
  return np.convolve(hist, kernel, mode='same')

##############################################################

def thresholding(image, threshold, typee='Binary', param1=0, param2=0):
  # A function to apply intensity thresholding to a grey-scale image
  # The thresholding could be simple binary thresholding or adaptive gaussian thresholding
  # If the type is not set to 'Binary' then the parameters for adaptive thresholdinf must
  # be used which are:
  #param1: local region size ( preferably an odd number)
  #param2: constant to be added to local mean
  if(typee.lower()=='binary'):
    ret, thresh= cv2.threshold(image,threshold,255,cv2.THRESH_BINARY_INV)
  else:
    thresh = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,param1,param2)
  return thresh

##############################################################

def peakinterp(interp_factor, hist, prominence_factor):
  #Given an intensity histogram of an image, this function increases the resolution of the histogram
  #by interpolation and then finds the sharp peaks in this histogram using find_peaks()
  #Interp factor controls the new resolution of the histogram
  #Prominence factor decides how much the targeted peaks stand out from the baseline of the spectrum
  resampled_pixel_space=np.linspace(0, interp_factor*len(hist)-1,interp_factor*len(hist))*(1/interp_factor)
  Original_pixel_space=np.linspace(0, len(hist)-1, len(hist))
  hist_interp = np.interp(resampled_pixel_space, Original_pixel_space, hist)
  peaks, properties = find_peaks(hist_interp, prominence=np.max(hist_interp)/prominence_factor, width=50)

  return(peaks,hist_interp, resampled_pixel_space, Original_pixel_space)

  ##############################################################

def findGradSignChange(hist_interp, resampled_pixel_space, Original_pixel_space):
    #Given an interpolated intensity histogram, this function finds the 1st derivative
    # of this histogram and outputs a vector of ones and zeros determining the sign
    # of the calculated derivative.
    # When the sign is +ve, the vector has 1
    # When the sign is -ve, the vector has 0
    hist_grad=np.gradient(hist_horizontal_smooth_interp)
    hist_grad_sign_change=np.where(hist_grad >= 0, 1, 0)
    return hist_grad_sign_change

   ##############################################################

def rle(ia):

        #A function which when given a sequence of binary values outputs the following:
        # 1) the start positions of a portion of repeated values in the sequence
        # 2) the length of the portion of repeated values
        #This will be useful in dealing with the vector representing the sign change of
        #1st derivative of image intensity histogram
       

        n = len(ia)
        if n == 0: 
            return (None, None, None)
        else:
            y = ia[1:] != ia[:-1]               # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)   # must include last element posi
            z = np.diff(np.append(-1, i))       # run lengths
            p = np.cumsum(np.append(0, z))[:-1] # positions
            return(z, p, ia[i])
 ##############################################################

def cutPositions(runlengths, startpositions, values, threshold,interp_factor):
  #Give a vector of ones and zeroes representing the sign change of 1st deriv. of
  # a histogram, this function smoothes out the abrupt changes in gradient sign
  # which might be an artifact of the gradient calculation.

  # This function also gives an estimation of the possible cutting locations to
  # extract lines

  viable_index=0
  for i in range(len(runlengths)):
    current_length=runlengths[i]
    if(current_length<threshold):
      values[i]=values[viable_index]
    viable_index=i

  new_hist=[]
  for i in range(len(startpositions)):
    if(values[i]):
      new_hist+=np.ones(runlengths[i]-1).tolist()
    else:
      new_hist+=np.zeros(runlengths[i]-1).tolist()

  cutpos=[]
  for i in range(1,len(startpositions)):
    last=values[i-1]
    current=values[i]
    if((last==0 and current==1)):
      cutpos.append(startpositions[i])
    elif((last==1 and i==1)):
      cutpos.append(0)


  return (cutpos, new_hist)

######################################################
def optimalThreshold(cutpos, runlengths, startpositions, values, new_hist, peaks, init_threshold, interp_factor):

  #when removing noise from the gradient sign vector prior to determining the cut locations, we use a threshold
  #value on the run lengths of ones and zeros.
  #An optimal value of the threshold is the value which when used gives us as many cut locations as detected peaks
  # in the original histogram
  while((len(cutpos)!= len(peaks))):
      init_threshold=init_threshold+interp_factor
      (cutpos, new_hist)=cutPositions(runlengths, startpositions, values, init_threshold,interp_factor)

  (cutpos, new_hist)=cutPositions(runlengths, startpositions, values, np.abs(init_threshold-interp_factor),interp_factor)
  cutpos=np.array(cutpos)/interp_factor
  
  return (cutpos, new_hist)

###################################################

def cropImageToLines(cutpos, image, direction='H'):
  (w,h)=image.shape
  cropped_images=[]
  if(direction=='H'):
    for i in range(len(cutpos)):
      currentpos=cutpos[i]
      lastpos=cutpos[i-1]
      cropped_images.append(image[lastpos:currentpos-1,0:h-1])
  else:
    for i in range(len(cutpos)):
      currentpos=cutpos[i]
      lastpos=cutpos[i-1]
      cropped_images.append(image[0:w-1, lastpos:currentpos-1])

  return cropped_images
import os
import cv2
import numpy as np

class LineSegmenter:
    def __init__(self, input_dir, line_dir, word_dir,
                 threshold_val=240, init_threshold=50, interp_factor=100):
        self.input_dir = input_dir
        self.line_dir = line_dir
        self.word_dir = word_dir
        self.threshold_val = threshold_val
        self.init_threshold = init_threshold
        self.interp_factor = interp_factor

        os.makedirs(self.line_dir, exist_ok=True)
        os.makedirs(self.word_dir, exist_ok=True)

    def segment_all(self, filenames):
        for idx, fname in enumerate(filenames):
            image = cv2.imread(os.path.join(self.input_dir, fname), 0)
            if image is None:
                print(f"[ERROR] Couldn't read {fname}")
                continue

            base_name = os.path.splitext(fname)[0]
            lines = self._segment_image(image)

            for i, line in enumerate(lines):
                line_name = f"{base_name}_{i}.tif"
                line_path = os.path.join(self.line_dir, line_name)
                cv2.imwrite(line_path, line)
                self._segment_words(line, line_name)

    def _segment_image(self, image):
        thresh = thresholding(image, self.threshold_val, typee='Binary', param1=0, param2=0)
        hist = directionalHistogram(thresh)
        hist_smooth = smoothHist(hist, 17)

        peaks, hist_interp, resampled_space, original_space = peakinterp(self.interp_factor, hist_smooth, 8)
        grad_changes = findGradSignChange(hist_interp, resampled_space, original_space)
        
        runlengths, startpositions, values = rle(grad_changes)
        cutpos, _ = cutPositions(runlengths, startpositions, values, self.init_threshold, self.interp_factor)
        cutpos, _ = optimalThreshold(cutpos, runlengths, startpositions, values, hist_smooth, peaks, 50, 100)

        return cropImageToLines(cutpos.astype(int), thresh)

    def _segment_words(self, line_img, filename):
        h, w = line_img.shape
        hist_v = directionalHistogram(line_img, direction='V')
        zero_sites = np.where(np.asarray(hist_v) == 0)[0]

        sequences, sequence_start = [], 0
        for i in range(1, len(zero_sites)):
            last_zero, curr_zero = zero_sites[i - 1], zero_sites[i]
            if curr_zero != last_zero + 1:
                sequences.append([sequence_start, last_zero])
                sequence_start = curr_zero
            if curr_zero == last_zero + 1 and i == len(zero_sites) - 1:
                sequences.append([sequence_start, curr_zero])

        sequence_lengths = [end - start + 1 for start, end in sequences]
        avg_seq_len = np.mean(sequence_lengths[1:-1]) if len(sequence_lengths) > 2 else 0
        overlap_factor = 0.75 * avg_seq_len

        viable_seqs = []
        for i, length in enumerate(sequence_lengths):
            if length >= avg_seq_len - overlap_factor:
                viable_seqs.extend(sequences[i])

        if viable_seqs and viable_seqs[0] != 0:
            viable_seqs = [0] + viable_seqs
        viable_seqs.append(-1)

        words = cropLineToWords(viable_seqs, line_img)[0]

        count = 0
        for i, word_img in enumerate(words):
            if np.sum(word_img) > 0:
                count += 1
                word_filename = f"{filename.replace('.tif','')}_word{count}.tif"
                cv2.imwrite(os.path.join(self.word_dir, word_filename), word_img)
                with open(os.path.join(self.word_dir, word_filename.replace('.tif', '.txt')), 'w') as f:
                    f.write('')


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
