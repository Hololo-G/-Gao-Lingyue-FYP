import os
import SimpleITK as sitk
import numpy as np
from PIL import Image as im

'''
   1st step: create a new file folder 'Dataset_corrected' to save processed images.
'''
def correct_data_set(path, o_path, normalized_folder='Dataset_normalized'):
    for sub_dir in ['Testing', 'Training', 'Validation']:
        sub_dir_path = os.path.join(path, sub_dir)
        o_sub_dir_path = os.path.join(o_path, sub_dir)
        for tumor_folder in ['glioma', 'meningioma', 'notumor', 'pituitary']:
            tumor_folder_path = os.path.join(sub_dir_path, tumor_folder)
            o_folder_path = os.path.join(o_sub_dir_path, tumor_folder)
            for i, image_file in enumerate(sorted(os.listdir(tumor_folder_path))):
                image_data_path = os.path.join(tumor_folder_path, image_file)
                new_name = f"{sub_dir[:2].lower()}-{tumor_folder[:2].lower()}_{i + 1:04d}.jpg"

                # Correct bias and get the corrected image path
                corrected_image_path = correct_bias(image_data_path, os.path.join(o_folder_path, new_name))

                if corrected_image_path is not None:
                    print(f"Processing image: {corrected_image_path}")
                    # Normalize the corrected image and save it in the specified folder
                    max_min_normalize(corrected_image_path, output_folder=normalized_folder)
'''
   2nd step: bias field correction to correct possible uneven brightness in the MRI image 
             caused by external factors
'''
def correct_bias(in_file, out_file, image_type=sitk.sitkFloat64):
    """
    Corrects the bias using SimpleITK N4BiasFieldCorrection.
    :param in_file: The path of input image
    :param out_file: The path to store the corrected image
    :return: The path to save image
    """
    try:
        input_image = sitk.ReadImage(in_file, image_type)
        output_image = sitk.N4BiasFieldCorrection(input_image, input_image > 0)
        output_image = sitk.GetArrayFromImage(output_image)
        output_image = im.fromarray(output_image.astype(np.uint8))

        # Ensure the output directory exists
        os.makedirs(os.path.dirname(out_file), exist_ok=True)

        return output_image.save(out_file)
    except Exception as e:
        # Handle any exceptions that may occur during SimpleITK processing
        print(f"Error during SimpleITK processing: {str(e)}")
        return None
'''
   3rd step: max_min normalization to scale the values of a feature to a specific range
'''
def max_min_normalize(image_path, output_folder='Dataset_normalized'):
    try:
        # Open the image
        image = im.open(image_path)

        # Convert PIL image to NP array
        image_array = np.array(image)

        # Calculate min and max from the NumPy array
        data_min = np.min(image_array)
        data_max = np.max(image_array)

        # Normalization
        normalized = (image_array - data_min) / (data_max - data_min)

        # Convert the normalized array back to PIL image
        normalized_image = im.fromarray((normalized * 255).astype(np.uint8))

        # Ensure the output directory exists
        os.makedirs(output_folder, exist_ok=True)

        # Save the normalized image in the specified folder
        normalized_image_path = os.path.join(output_folder, os.path.basename(image_path).replace(".jpg", "_normalized.jpg"))
        normalized_image.save(normalized_image_path)

        return normalized_image_path
    except Exception as e:
        # Handle any exceptions that may occur during normalization
        print(f"Error during normalization: {str(e)}")
        return None

# 调用示例
correct_data_set(r'D:\SBC\FYP\Dataset_original', r'D:\SBC\FYP\Dataset_corrected')

# Bias correction and max-min normalization have been done 2024.3.5 19:00

'''
   4th step: resize the images to suit the input format of VGG-16
'''
from PIL import Image
import os

def data_resize(folder_path, target_size=(224, 224)):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                # Print the current processing image path
                print(f"Processing: {image_path}")
                # Open the image
                image = Image.open(image_path)
                # Resize the image
                resized_image = image.resize(target_size)
                # Save the resized image, overwriting the original file
                resized_image.save(image_path)
if __name__ == "__main__":
    # Specify the path to the folder containing the images
    folder_path = r'D:\SBC\FYP\Dataset_corrected'
    # Specify the target size for resizing
    target_size = (224, 224)
    # Resize images in the specified folder
    data_resize(folder_path, target_size)

# Image resize has been done 2024.3.5 19:38

'''
   5th step: convert images from greyscale channel to RGB channel
'''
from PIL import Image
import os

def convert_to_rgb(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                # Print the current processing image path
                print(f"Processing: {image_path}")
                # Open the image
                image = Image.open(image_path)
                # Convert to RGB
                rgb_image = Image.merge('RGB', (image, image, image))
                # Save the resized image, overwriting the original file
                rgb_image.save(image_path)
if __name__ == "__main__":
    # Specify the path to the folder containing the images
    folder_path = r'D:\SBC\FYP\Dataset_corrected'
    # Resize images in the specified folder
    convert_to_rgb(folder_path)

# Greyscale to RGB conversion has been done 2024.3.5 19:52

'''
   6th step: Contrast enhancement
'''
from PIL import ImageEnhance
import os
from PIL import Image

def enhance_contrast(folder_path, enhancement_factor, save_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".jpg"):
                image_path = os.path.join(root, file)
                # Print the current processing image path
                print(f"Processing: {image_path}")
                # Open the image
                image = Image.open(image_path)
                # Enhance contrast
                enhancer = ImageEnhance.Contrast(image)
                contrast_image = enhancer.enhance(enhancement_factor)
                # Get the relative path within the folder_path
                relative_path = os.path.relpath(image_path, folder_path)
                # Construct the save path
                save_dir = os.path.join(save_path, os.path.dirname(relative_path))
                os.makedirs(save_dir, exist_ok=True)  # Create directories if they don't exist
                save_file = os.path.join(save_path, relative_path)
                # Save the enhanced image
                contrast_image.save(save_file)

if __name__ == "__main__":
    # Specify the path to the folder containing the images
    folder_path = r'D:\SBC\FYP\Dataset_corrected'
    # Specify the path to save the enhanced images
    save_path = r'D:\SBC\FYP\Dataset_enhanced'
    # Enhancement factor (1.0 means no change)
    enhancement_factor = 1.5  # You can adjust this value as needed
    # Enhance contrast in the specified folder
    enhance_contrast(folder_path, enhancement_factor, save_path)

