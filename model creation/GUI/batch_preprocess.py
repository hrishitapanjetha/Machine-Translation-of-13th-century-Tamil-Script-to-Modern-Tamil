import os
from image_functions import preprocess
import cv2
def wasiq_preprocess_image(image):
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # smoothen the image
    denoised = cv2.GaussianBlur(binary, (1, 1), 0)

    # Noise Reduction using Gaussian Blur
    # denoised = cv2.GaussianBlur(binary, (1, 1), 0)
    denoised = cv2.medianBlur(binary, 3)
    
    return denoised
def preprocess_images(input_directory, output_directory):

    for filename in os.listdir(directory):
        if filename.endswith(".jpg"):
            print(f"Processing {filename}")
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            processed_image = wasiq_preprocess_image(image)
            cv2.imwrite(f"{output_directory}/{filename}", processed_image)

directory = "./Dataset/all_images"
save_path = "./Dataset/wasiq_preprocessed_images"
os.makedirs(save_path, exist_ok=True)
# preprocess all images in the directory
preprocess_images(directory, save_path)
