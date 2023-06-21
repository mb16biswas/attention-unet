# Image Segmentation of Metallographic Images

**Microstructural segmentation using a union of attention guided U-Net models with different color transformed image**

Find the original paper [here](https://www.nature.com/articles/s41598-023-32318-9#Sec4).

![Overall_Pipeline](https://github.com/mb16biswas/attention-unet/assets/64213233/a2268d6a-2b83-42b9-959d-279703348ca9)

```
@article{biswas2023microstructural,
  title={Microstructural segmentation using a union of attention guided U-Net models with different color transformed images},
  author={Biswas, Momojit and Pramanik, Rishav and Sen, Shibaprasad and Sinitca, Aleksandr and Kaplun, Dmitry and Sarkar, Ram},
  journal={Scientific Reports},
  volume={13},
  number={1},
  pages={5737},
  year={2023},
  publisher={Nature Publishing Group UK London}
}

```


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
python3 main.py  --images_path "images_path" --masks_path "masks_path"
```

Available arguments:
- `--images_path`: Path where the images folder is stored. Default = ./
- `--masks_path`: Path where the masks folder is stored. Default = ./
- `--epochs`: Number of epochs of training. Default = 250
- `--lr`: Learning rate for training. Default = 0.001
- `--batch`: Batch Size for Mini Batch Training. Default = 4
- `--n_splits`: Number of folds for training. Default= 6
- `--show`: Showing the comparison among original, ground-truth and predicted images. Default = False



3. The increase_data.py is for increasing the datasize.

```
python3 increase_data.py --images_path "images_path" --masks_path "masks_path" --target_folder "target_folder"
```

Available arguments:
- `--images_path`: Path where the images folder is stored. Default = ./
- `--masks_path`: Path where the masks folder is stored. Default = ./
- `--target_folder`: arget folder where the images folder and the maskes folder are stored. Default = ./
- `--n`: Increase the data n number of times. Default = 6

