import image_utils, text_utils
from PIL import Image
import re, json, nltk, itertools, spacy, difflib, math
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


def extract_text_from_image(image_path):
    image = Image.open(image_path)
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
    - document_type (e.g., Aadhaar Card-12digit, Voter ID-10digit, Driving License-15 digit, Debit Card-16 digit, Credit Card-16 digit, PAN Card-10 digit, Passport-8 digit)
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


if __name__ == '__main__':
    image_path = './Dummy/image.png'
    original = extract_text_from_image(image_path)
    result = get_formatted_text_info(original)
    print(result)
    

