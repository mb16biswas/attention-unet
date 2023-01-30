import os
import cv2
from tqdm import tqdm



"""
increases the dataset size by the 4 times using the same image-augmentation
in the actual image and the mask

"""

def augment_data4(images, masks,path,W=256, H =256):

    """
      images: array of file paths for the image
      mask: array of file paths for the image
      path: base folder path

    """
    id = 0

    for x, y in tqdm(zip(images, masks), total=len(images)):

        x = cv2.imread(x, 1)
        y = cv2.imread(y, 1)



        x1 = cv2.flip(x, 0 )
        y1 = cv2.flip(y, 0 )


        x2 = cv2.flip(x, 1 )
        y2 = cv2.flip(y, 1 )


        x3 = cv2.flip(x, -1 )
        y3 = cv2.flip(y, -1 )


        save_images = [x, x1, x2, x3]
        save_masks =  [y, y1, y2, y3]



        """ Saving the image and mask. """
        idx = 0
        for i, m in zip(save_images, save_masks):

            tmp_img_name =  "image" + str(id) + str(idx) + ".png"
            tmp_mask_name = "mask" +  str(id)+  str(idx) + ".png"

            image_path = os.path.join(path, "image", tmp_img_name)
            mask_path = os.path.join(path, "mask", tmp_mask_name)

            cv2.imwrite(image_path, i)
            cv2.imwrite(mask_path, m)

            idx += 1
        id += 1

