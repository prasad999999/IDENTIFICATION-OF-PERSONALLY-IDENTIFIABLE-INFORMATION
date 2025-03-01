import image_utils
from PIL import Image
import re, json, nltk, itertools, spacy, difflib, math, cv2, easyocr
import numpy as np
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
nltk_resources = ["punkt", "maxent_ne_chunker_tab", "words", "averaged_perceptron_tagger", "stopwords"]
import google.generativeai as genai
from dotenv import load_dotenv
import os

for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

load_dotenv()

# Configure API with Free Google Gemini Model
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing. Check your .env file.")

genai.configure(api_key=api_key)


def extract_text_from_image(image):
    text_content = image_utils.scan_image_for_text(image)

    
    image_text_unmodified = text_content["unmodified"]
    image_text_auto_rotate = text_content["auto_rotate"]
    image_text_grayscaled = text_content["grayscaled"]
    image_text_monochromed = text_content["monochromed"]
    image_text_mean_threshold = text_content["mean_threshold"]
    image_text_gaussian_threshold = text_content["gaussian_threshold"]
    image_text_deskewed_1 = text_content["deskewed_1"]
    image_text_deskewed_2 = text_content["deskewed_2"]
    image_text_deskewed_3 = text_content["deskewed_3"]

    # unmodified_words = text_utils.string_tokenizer(image_text_unmodified)
    # grayscaled = text_utils.string_tokenizer(image_text_auto_rotate)
    # auto_rotate = text_utils.string_tokenizer(image_text_grayscaled)
    # monochromed = text_utils.string_tokenizer(image_text_monochromed)
    # mean_threshold = text_utils.string_tokenizer(image_text_mean_threshold)
    # gaussian_threshold = text_utils.string_tokenizer(image_text_gaussian_threshold)
    # deskewed_1 = text_utils.string_tokenizer(image_text_deskewed_1)
    # deskewed_2 = text_utils.string_tokenizer(image_text_deskewed_2)
    # deskewed_3 = text_utils.string_tokenizer(image_text_deskewed_3)

    original = image_text_unmodified + "\n" + image_text_auto_rotate + "\n" + image_text_grayscaled + "\n" + image_text_monochromed + "\n" + image_text_mean_threshold + "\n" + image_text_gaussian_threshold + "\n" + image_text_deskewed_1 + "\n" + image_text_deskewed_2 + "\n" +  image_text_deskewed_3

    print('\n')
    print(original)
    print('\n')

    # intelligible = unmodified_words + grayscaled + auto_rotate + monochromed + mean_threshold + gaussian_threshold + deskewed_1 + deskewed_2 + deskewed_3

    return original

def get_formatted_text_info(text):
    """Extracts formatted information from the given text using Google Gemini API."""
    
    prompt = f"""
    Extract the following details from the given text and return a JSON response:
    - document_type (e.g., Aadhaar Card (12digit), Voter ID (10 digit), Driving License (15 digit), Debit Card (16 digit), Credit Card (16 digit), PAN Card (10 digit), Passport (8 digit))
    - name
    - country
    - Document id (if applicable)
    - email
    - phone_no
    - address
    - DOB
    - Gender
    - Expiry Date(if applicable)

    Text: {text}
    
    Return the response in JSON format.
    """

    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")  # ✅ Use the correct model
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return {"error": str(e)}

def get_json_data(response):
    cleaned_result = re.sub(r"```json|\n```", "", result).strip()
    if cleaned_result and cleaned_result.strip():  # Ensure it's not empty
        try:
            result_json = json.loads(cleaned_result)  # Convert to Python dictionary
            print("Parsed JSON:", result_json)
            return result_json
        except json.JSONDecodeError as e:
            print("Error parsing JSON:", e)
            result_json = {}  # Default to an empty dictionary
            return result_json
    else:
        print("Error: Received empty or invalid JSON response")
        result_json = {}
        return result_json

def mask_image(image_path, result_json):
    image = cv2.imread(image_path)
    
    # Sensitive information to mask
    sensitive_info = [
        result_json.get("name"), 
        result_json.get("Document id"), 
        result_json.get("phone_no"), 
        result_json.get("address"), 
        result_json.get("DOB"), 
        result_json.get("Expiry Date")
    ]
    sensitive_info = [str(info) for info in sensitive_info if info is not None]
    
    # Break sensitive info into smaller chunks
    chunks = []
    for info in sensitive_info:
        # Add the whole string
        chunks.append(info)
        # Also add individual parts split by spaces
        parts = info.split()
        for part in parts:
            if len(part) >= 3:  # Only consider chunks of reasonable length
                chunks.append(part)

    reader = easyocr.Reader(['en'])
    ocr_results = reader.readtext(image)

    for (bbox, text, prob) in ocr_results:
        # Check if the OCR text matches or is contained in any sensitive info
        detected_text = text.strip()
        
        # Skip very short text (likely to cause false positives)
        if len(detected_text) < 3:
            continue
            
        for chunk in chunks:
            # Case-insensitive comparison
            if (detected_text.lower() in chunk.lower() or 
                chunk.lower() in detected_text.lower()):
                
                # Get bounding box coordinates
                (top_left, top_right, bottom_right, bottom_left) = bbox
                x_min = int(top_left[0])
                y_min = int(top_left[1])
                x_max = int(bottom_right[0])
                y_max = int(bottom_right[1])
                
                # Apply a blur mask over the detected text
                roi = image[y_min:y_max, x_min:x_max]

                # Apply Gaussian Blur to the ROI
                blurred_roi = cv2.GaussianBlur(roi, (15, 15), 30)

                # Replace the original ROI with the blurred ROI
                image[y_min:y_max, x_min:x_max] = blurred_roi
                break
                
    cv2.imwrite("masked_document.jpg", image)
    print("Masked Image Saved")


if __name__ == '__main__':
    image_path = './Dummy/image.png'
    image = Image.open(image_path)
    original = extract_text_from_image(image)
    result = get_formatted_text_info(original)
    json_data = get_json_data(result)
    mask_image(image_path, json_data)
    

