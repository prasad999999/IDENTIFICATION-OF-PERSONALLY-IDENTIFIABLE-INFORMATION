import image_utils, text_utils
from PIL import Image

if __name__ == '__main__':
    image_path = "image.png"
    image = Image.open(image_path)
    text_content = image_utils.scan_image_for_text(image)

    unmodified_words = text_utils.string_tokenizer(text_content["unmodified"])
    grayscaled = text_utils.string_tokenizer(text_content["auto_rotate"])
    auto_rotate = text_utils.string_tokenizer(text_content["grayscaled"])
    monochromed = text_utils.string_tokenizer(text_content["monochromed"])
    mean_threshold = text_utils.string_tokenizer(text_content["mean_threshold"])
    gaussian_threshold = text_utils.string_tokenizer(text_content["gaussian_threshold"])
    deskewed_1 = text_utils.string_tokenizer(text_content["deskewed_1"])
    deskewed_2 = text_utils.string_tokenizer(text_content["deskewed_2"])
    deskewed_3 = text_utils.string_tokenizer(text_content["deskewed_3"])

    print(unmodified_words, '\n', grayscaled, '\n', auto_rotate, '\n', monochromed, '\n', mean_threshold, '\n'
          , gaussian_threshold, '\n', deskewed_1, '\n', deskewed_2, '\n', deskewed_3,'\n')
    

