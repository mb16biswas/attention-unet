from utils.cross_val_results.k_fold import k_fold
import argparse
import os




parser = argparse.ArgumentParser()

parser.add_argument('--images_path', type=str, default = './',
                    help='Path where the images folder is stored')

parser.add_argument('--masks_path', type=str, default = './',
                    help='Path where the masks folder is stored')

parser.add_argument('--epochs', type=int, default = 250,
                    help='Number of Epochs for training')

parser.add_argument('--batch', type=int, default = 4,
                    help='Batch Size for Mini Batch Training')

parser.add_argument('--n_splits', type=int, default = 6,
                    help='Number of folds for training')

parser.add_argument('--lr', type=float, default = 1e-3,
                    help='Learning rate for training')

parser.add_argument('--show', type=bool, default = False,
                    help='Showing the comparison among original, ground-truth and predicted images')

args = parser.parse_args()




path_img_agu =  args.images_path
img = []

for f in sorted(os.listdir(path_img_agu)):
  img.append(os.path.join(path_img_agu ,f))



path_label_agu =  args.masks_path
labels = []

for f in sorted(os.listdir(path_label_agu)):
  labels.append(os.path.join(path_label_agu,f))




for i in range(0,args.n_splits):

    k_fold(img,labels,i ,batch = args.batch, epochs = args.epochs , lr = args.lr , n_splits = args.n_splits ,show = args.show)



