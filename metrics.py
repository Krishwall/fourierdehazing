import cv2
from skimage.metrics import structural_similarity
import numpy as np
import os

def metrics(img1_path, img2_path):
    # Load the original and distorted images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        raise ValueError("One of the images could not be loaded.")

    # Calculate PSNR (Peak Signal-to-Noise Ratio)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        psnr = 100
    else:
        pixel_max = 255.0
        psnr = 20 * np.log10(pixel_max / np.sqrt(mse))

    gray_image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM
    ssim = structural_similarity(gray_image1, gray_image2)
    return psnr, ssim

source = r"D:\fourierdehazing\dataset\hazy_images"
dest = r"D:\fourierdehazing\dataset\dehazed_images"
images = os.listdir(source)

psnr_total = 0
ssim_total = 0
count = 0

for image in images:
    try:
        source_path = os.path.join(source, image)
        dest_path = os.path.join(dest, image[:-4] + "_dehazed" + image[-4:])
        
        if not os.path.exists(dest_path):
            print(f"Dehazed image not found for: {image}")
            continue

        psnr, ssim = metrics(source_path, dest_path)
        psnr_total += psnr
        ssim_total += ssim
        count += 1
    except Exception as e:
        print(f"Error processing {image}: {e}")

if count > 0:
    psnr = psnr_total / count
    ssim = ssim_total / count
    print("PSNR:", psnr)
    print("SSIM:", ssim)
else:
    print("No valid image pairs found for comparison.")