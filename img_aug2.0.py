from imgaug import augmenters as iaa
import numpy as np
from PIL import Image
import os
import glob
import time

# Start Operation Time
start = time.time()
# Intitialise a few things here....
counter = 0
ttl_img = 0
lbl_inc = 0


# Ask user for input regarding what to use as root directory for folders containing image groups to be augmented
root = input("Enter root directory of image classification folders: ")
# Ask user for input regarding how many augmented images to output for each input image.
iterations = int(input("How many augmented copies do you want for each input image? "))
# Get a list of subdirectory names in order to find images for augmentation in batches
dir_list = next(os.walk(root))[1]
# Append subdirectory name onto root directory name to give an absolute path to the files in the list
dir_list = [root + directory + "/*" for directory in dir_list]

# Setting up Augmentation sequence to apply to source image
seq = iaa.Sequential([

    # Training Sets
    # Translation of image in X & Y axis by random pixel amount within bounds 45% of the time
    iaa.Sometimes(0.45, iaa.Affine(translate_px={"x": (-4, 4), "y": (-4, 4)})),
    # Rotation of image by random amount within bounds 45% of the time
    iaa.Sometimes(0.45, iaa.Affine(rotate=(-2, 2))),
    # Scales image between bounds 5% of the time
    iaa.Sometimes(0.05, iaa.Affine(scale={"x": (0.95, 1.05), "y": (0.95, 1.05)})),
    # Add Blur to image 25% of the time
    iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.0))),], random_order=True)

    # Validation Sets
    # Translation of image in X & Y axis by random pixel amount within bounds 45% of the time
    # iaa.Sometimes(0.45, iaa.Affine(translate_px={"x": (-5, 5), "y": (-5, 5)})),
    # Rotation of image by random amount within bounds 45% of the time
    # iaa.Sometimes(0.45, iaa.Affine(rotate=(-3, 3))),
    # Scales image between bounds 5% of the time
    # iaa.Sometimes(0.05, iaa.Affine(scale={"x": (0.92, 1.08), "y": (0.92, 1.08)})),
    # Add Blur to image 25% of the time
    # iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 1.0))),], random_order=True)

# for each directory found above, find and import all images in directory for augmentation
for directory in dir_list:
    # Increment counter to record how many images are being output
    ttl_img += lbl_inc
    # Reset label increment so each new folders images starts from 1
    lbl_inc = 0
    # python cannot deal with backslashes...(char(92)....so i replace them with forward slashes for the file directory
    clean_dir = directory.replace(chr(92), '/')
    # Finding all files in given directory folder
    source = glob.glob(clean_dir)
    # loading all files into an array of matrices so that we can work with them
    src_img = np.array([np.array(Image.open(src_filename)) for src_filename in source])
    # print("Image size is(d x w x h):", src_img.shape)

    for batch_idk in range(iterations):  # number of batches to do, 2 would double image library, 3 triple etc etc
        # Augment each element of the image matrices given the previously declared sequence and operations (Line21)
        aug_img = seq.augment_images(src_img)
        # Get number of images in the array by getting its depth
        shape = aug_img.shape
        # Assigning 1st dimension of array (depth) to variable for use in the image saving loop
        img_num = shape[0]
        print("number of images is", img_num)
        # Loop to save each image in data set to file if there is multiple source images
        for count in range(img_num):
            lbl_inc += 1
            # Cut a slice of 3D array to isolate one image for saving
            single_img = aug_img[count][:][:]
            # Save data from array as an image ready for export
            des_img = Image.fromarray(single_img, "L")
            # Prepare the directory path for file save
            des_path = clean_dir.replace("*", '{}')
            # append counter to file path to enable saving multiple images without them being overwritten
            des_filename = des_path.format(lbl_inc)
            # print(des_filename)
            # Save image to directory where original was found
            des_img.save(des_filename + '.jpeg', 'jpeg')

print("Total Images Output: ", ttl_img)
print("Time Taken %0.2fs" % (time.time()-start))
print("Augmentation Completed!")
