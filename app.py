import image_utils, text_utils
from PIL import Image
import re, json, nltk, itertools, spacy, difflib, math
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.corpus import stopwords
nltk_resources = ["punkt", "maxent_ne_chunker_tab", "words", "averaged_perceptron_tagger", "stopwords"]

for resource in nltk_resources:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource)

if __name__ == '__main__':
    image_path = "Dummy/image.png"
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

    unmodified_words = text_utils.string_tokenizer(image_text_unmodified)
    grayscaled = text_utils.string_tokenizer(image_text_auto_rotate)
    auto_rotate = text_utils.string_tokenizer(image_text_grayscaled)
    monochromed = text_utils.string_tokenizer(image_text_monochromed)
    mean_threshold = text_utils.string_tokenizer(image_text_mean_threshold)
    gaussian_threshold = text_utils.string_tokenizer(image_text_gaussian_threshold)
    deskewed_1 = text_utils.string_tokenizer(image_text_deskewed_1)
    deskewed_2 = text_utils.string_tokenizer(image_text_deskewed_2)
    deskewed_3 = text_utils.string_tokenizer(image_text_deskewed_3)

    original = image_text_unmodified + "\n" + image_text_auto_rotate + "\n" + image_text_grayscaled + "\n" + image_text_monochromed + "\n" + image_text_mean_threshold + "\n" + image_text_gaussian_threshold + "\n" + image_text_deskewed_1 + "\n" + image_text_deskewed_2 + "\n" +  image_text_deskewed_3

    intelligible = unmodified_words + grayscaled + auto_rotate + monochromed + mean_threshold + gaussian_threshold + deskewed_1 + deskewed_2 + deskewed_3

    print(original)
    print('\n\n')
    print(intelligible)
    

