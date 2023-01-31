import numpy as np
from tf.keras.metrics import MeanIoU
from evaluation.helpers import pred


def mean_iou_score(model,X_val,y_val,n_classes ):

    y_acc , y_pred = pred(X_val,y_val,model , len(y_val))
    IOU_ref = MeanIoU(num_classes=n_classes)
    IOU_ref.update_state(y_acc , y_pred)


    values = np.array(IOU_ref.get_weights()).reshape(n_classes,n_classes)


    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0]+ values[2,0]+ values[3,0] + values[4,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[0,2]+ values[1,2]+ values[3,2]  +values[4,2] )
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3]+ values[1,3]+ values[2,3] + values[4,3])
    class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2]  +values[4,3] + values[0,4]+ values[1,4]+ values[2,4] + values[3,4])


    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)
    print("IoU for class4 is: ", class4_IoU)
    print("IoU for class5 is: ", class5_IoU)
    print("IoU for class5 is mean: ", (class1_IoU + class2_IoU + class3_IoU + class4_IoU + class5_IoU)/4)




def mean_iou_score_ensemble(y_preds,y_actual,n_classes = 5 ):

    y1 = []

    for i in range(0,len(y_preds)):

        y1.append(y_preds[i].reshape(256,256,1))

    y1 = np.array(y1)

    IOU_ref = MeanIoU(num_classes=n_classes)
    IOU_ref.update_state(y_actual , y1)


    values = np.array(IOU_ref.get_weights()).reshape(n_classes,n_classes)



    class1_IoU = values[0,0]/(values[0,0] + values[0,1] + values[0,2] + values[0,3] + values[0,4] + values[1,0]+ values[2,0]+ values[3,0] + values[4,0])
    class2_IoU = values[1,1]/(values[1,1] + values[1,0] + values[1,2] + values[1,3] + values[1,4] + values[0,1]+ values[2,1]+ values[3,1] + values[4,1])
    class3_IoU = values[2,2]/(values[2,2] + values[2,0] + values[2,1] + values[2,3] + values[2,4]+ values[0,2]+ values[1,2]+ values[3,2]  +values[4,2] )
    class4_IoU = values[3,3]/(values[3,3] + values[3,0] + values[3,1] + values[3,2] + values[3,4] + values[0,3]+ values[1,3]+ values[2,3] + values[4,3])
    class5_IoU = values[4,4]/(values[4,4] + values[4,0] + values[4,1] + values[4,2]  +values[4,3] + values[0,4]+ values[1,4]+ values[2,4] + values[3,4])


    print("IoU for class1 is: ", class1_IoU)
    print("IoU for class2 is: ", class2_IoU)
    print("IoU for class3 is: ", class3_IoU)
    print("IoU for class4 is: ", class4_IoU)
    print("IoU for class5 is: ", class5_IoU)
    print("IoU for class5 is mean: ", (class1_IoU + class2_IoU + class3_IoU + class4_IoU + class5_IoU)/4)
