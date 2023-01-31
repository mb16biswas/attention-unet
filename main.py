from utils.cross_val_results import k_fold

x = "./filepath/x"
y = "./filepath/y"

for i in range(0,6):

    k_fold(x,y,i)