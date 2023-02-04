import os
from augmentation.augment_data10 import augment_data10
from augmentation.augment_data6 import augment_data6
from augmentation.augment_data4 import augment_data4
import argparse


def new_dataset_len(path_name):


    print("final total images .. " , len(os.listdir(os.path.join(path_name, "image"))))
    print("final total masks .. " , len(os.listdir(os.path.join(path_name, "mask"))))



parser = argparse.ArgumentParser()

parser.add_argument('--images_path', type=str, default = './',
                    help='Path where the images folder is stored')

parser.add_argument('--masks_path', type=str, default = './',
                    help='Path where the masks folder is stored')

parser.add_argument('--target_folder', type=str, default = './',
                    help='Target folder where the images folder and the maskes folder are stored')



parser.add_argument('--n', type=int, default = 6,
                    help='Increase the data n number of times')



args = parser.parse_args()


path_img =  args.images_path
path_label = args.masks_path
n = args.n
target = args.target_folder


img = []
for f in sorted(os.listdir(path_img)):
  img.append(os.path.join(path_img ,f))



labels = []
for f in sorted(os.listdir(path_label)):
  labels.append(os.path.join(path_label,f))



print("initial total masks .. " , len(labels))
print("initial total images .. " , len(img))



if(n == 4 ):


    augment_data4(img, labels,target )
    new_dataset_len(target)

elif(n == 6):

    augment_data6(img, labels, target )
    new_dataset_len(target)

elif(n == 10):

    augment_data10(img, labels, target )
    new_dataset_len(target)

else:

    print("invalid ..")
