# Image Segmentation of Metallographic Images

**Microstructural segmentation using a union of attention guided U-Net models with different color transformed image**


# Dataset Link
[MetalDam](https://github.com/ari-dasci/OD-MetalDAM)


# Instructions to run the code
Required directory structure:
(Note: data contains subfolders of images and masks.)
```
+-- data
|   +-- images
|   |   +--image00
|   |   +--image01
|   |   +--image02
|   |   ...
|   +-- masks
|   |   +--mask00
|   |   +--mask01
|   |   +--mask02
|   |   ...
+-- main.py
```
1. Download the repository and install the required packages:
```
pip3 install -r requirements.txt
```
2. The main file is sufficient to run the experiments.
Then, run the code using linux terminal as follows:

```
python3 main.py --data_directory --images_path "images_path" --masks_path "masks_path"
```

Available arguments:
- `--images_path`: Path where the images folder is stored. Default = ./
- `--masks_path`: Path where the masks folder is stored. Default = ./
- `--epochs`: Number of epochs of training. Default = 250
- `--lr`: Learning rate for training. Default = 0.001
- `--batch`: Batch Size for Mini Batch Training. Default = 4
- `--n_splits`: Number of folds for training. Default= 6
- `--show`: Showing the comparison among original, ground-truth and predicted images. Default = False


```
3. The increase.py is for increasing the datasize.

```

```
python3 main.py --images_path "images_path" --masks_path "masks_path" --target_folder "target_folder"
```

Available arguments:
- `--images_path`: Path where the images folder is stored. Default = ./
- `--masks_path`: Path where the masks folder is stored. Default = ./
- `--target_folder`: arget folder where the images folder and the maskes folder are stored. Default = ./
- `--n`: Increase the data n number of times. Default = 6 


