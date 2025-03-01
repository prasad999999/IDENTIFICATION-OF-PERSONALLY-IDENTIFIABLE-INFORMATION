from PIL import Image
from skimage.transform import rotate
from deskew import determine_skew
import numpy as np
import cv2
import easyocr
import text_utils

def scan_image_for_text(image):
    reader = easyocr.Reader(['en'])  # Initialize EasyOCR reader
    image = np.array(image)
    
    def extract_text(img):
        results = reader.readtext(img, detail=0)
        return " ".join(results)
    
    try:
        image_text_unmodified = extract_text(image)
        # print(image_text_unmodified)
    except TypeError:
        print("Cannot open this file type.")
        return
    
    try:
        degrees_to_rotate = 180  # Default rotation
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        degrees_to_rotate = determine_skew(gray)
        rotated = rotate(image, degrees_to_rotate, resize=True) * 255
        image = rotated.astype(np.uint8)
        image_text_auto_rotate = extract_text(image)
        # print(image_text_auto_rotate)
    except:
        print("Couldn't auto-rotate image")
        image_text_auto_rotate = ""
    
    try:
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_text_grayscaled = extract_text(image_gray)
        # print(image_text_grayscaled)
    except:
        print("Couldn't grayscale image")
        image_text_grayscaled = ""
    
    try:
        _, image_mono = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        image_text_monochromed = extract_text(image_mono)
        # print(image_text_monochromed)
    except:
        print("Couldn't monochrome image")
        image_text_monochromed = ""
    
    try:
        image_mean = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
        image_text_mean_threshold = extract_text(image_mean)
        # print(image_text_mean_threshold)
    except:
        print("Couldn't apply mean threshold")
        image_text_mean_threshold = ""
    
    try:
        image_gaussian = cv2.adaptiveThreshold(image_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        image_text_gaussian_threshold = extract_text(image_gaussian)
        # print(image_text_gaussian_threshold)
    except:
        print("Couldn't apply gaussian threshold")
        image_text_gaussian_threshold = ""
    

    try:
        # 6a. Deskew one
        #print ("First attempt at de-skewing image and reading text")
        angle = determine_skew(image)
        rotated = rotate(image, angle, resize=True) * 255
        image = rotated.astype(np.uint8)
        image_text_deskewed_1 = extract_text(image)
        #cv2.imwrite("image6.png", image)

        # 6b. Deskew two
        #print ("Second attempt at de-skewing image and reading text")
        angle = determine_skew(image)
        rotated = rotate(image, angle, resize=True) * 255
        image = rotated.astype(np.uint8)
        image_text_deskewed_2 = extract_text(image)
        #cv2.imwrite("image7.png", image)

        # 6c. Deskew three
        #print ("Third attempt at de-skewing image and reading text")
        angle = determine_skew(image)
        rotated = rotate(image, angle, resize=True) * 255
        image = rotated.astype(np.uint8)
        image_text_deskewed_3 = extract_text(image)
        #cv2.imwrite("image8.png", image)
    except: 
        print ("Couldn't deskew image")
        image_text_deskewed_1 = ""
        image_text_deskewed_2 = ""
        image_text_deskewed_3 = ""
    

    return {
        "unmodified": image_text_unmodified,
        "auto_rotate": image_text_auto_rotate,
        "grayscaled": image_text_grayscaled,
        "monochromed": image_text_monochromed,
        "mean_threshold": image_text_mean_threshold,
        "gaussian_threshold": image_text_gaussian_threshold,
        "deskewed_1": image_text_deskewed_1,
        "deskewed_2": image_text_deskewed_2,
        "deskewed_3": image_text_deskewed_3,
    }

# image_path = "Dummy/image.png"
# image = Image.open(image_path)

# original,result = scan_image_for_text(image)
# print(original)