import numpy as np
from utils.dataprocessing.create_image import process_x, process_x2
from utils.dataprocessing.create_mask import process_y, process_y2
from utils.dataprocessing.further_processing import further_process
import segmentation_models_3D as sm
from utils.unet.build_unet import build_unet
from utils.evaluation.helpers import show_predictions,display
from utils.evaluation.mean_iou import mean_iou_score, mean_iou_score_ensemble




def k_fold(X,y,fold, batch , epochs , lr  , n_splits ,show ):

    print()
    print()
    print("..................................................................................................")
    print("..................................................................................................")
    print()
    print("fold " , fold + 1 )
    print()
    print("..................................................................................................")
    print("..................................................................................................")
    print()
    print()

    np.random.seed(42)  #default 42
    X = np.random.choice(X, size= len(X) , replace=False)
    np.random.seed(42)  #default 42
    y = np.random.choice(y, size= len(y)  , replace=False)

    X = np.array_split(X, n_splits)
    y = np.array_split(y, n_splits)

    l = []
    for i in range(0,n_splits):
        if(i != fold):
            l.append(i)



    X_train = np.concatenate([X[l[i]] for i in range(0,len(l))], axis=0)
    y_train = np.concatenate([y[l[i]] for i in range(0,len(l))], axis=0)

    np.random.seed(42)
    X_val = np.random.choice(X_train, size= int(len(X_train)*0.1) , replace=False)

    np.random.seed(42)
    y_val = np.random.choice(y_train, size= int(len(y_train)*0.1) , replace=False)


    X_test = X[fold]
    y_test = y[fold]

    X_train_image_normal = process_x2(X_train)
    y_train_image_normal  = process_y2(y_train, cat = True )

    X_train_image_HSV = process_x2(X_train , con = "HSV" )
    y_train_image_HSV  = process_y2(y_train, cat = True )

    X_train_image_YUV = process_x2(X_train, con = "YUV")
    y_train_image_YUV  = process_y2(y_train, cat = True )



    #val
    X_val_image_normal = process_x2(X_val)
    y_val_image_normal  = process_y2(y_val, cat = True )

    X_val_image_HSV = process_x2(X_val , con = "HSV" )
    y_val_image_HSV  = process_y2(y_val, cat = True )

    X_val_image_YUV = process_x2(X_val, con = "YUV")
    y_val_image_YUV  = process_y2(y_val, cat = True )



    #test
    X_test_image_normal = process_x(X_test)
    y_test_image_normal = process_y(y_test)

    X_test_image_HSV = process_x(X_test , con = "HSV" )
    y_test_image_HSV = process_y(y_test)

    X_test_image_YUV = process_x(X_test , con = "YUV")
    y_test_image_YUV = process_y(y_test)



    train_normal , val_normal = further_process(X_train_image_normal,y_train_image_normal,X_val_image_normal,y_val_image_normal)

    train_HSV , val_HSV = further_process(X_train_image_HSV,y_train_image_HSV,X_val_image_HSV,y_val_image_HSV)

    train_YUV , val_YUV = further_process(X_train_image_YUV,y_train_image_YUV,X_val_image_YUV,y_val_image_YUV)



    STEPS_PER_EPOCH = len(X_train)//batch
    VALIDATION_STEPS = len(X_val)//batch




    dice_loss = sm.losses.DiceLoss(class_weights=np.array([0.20, 0.20, 0.20, 0.20,0.20]))
    focal_loss = sm.losses.CategoricalFocalLoss()
    total_loss = dice_loss + (1 * focal_loss)
    metrics = ["accuracy",sm.metrics.IOUScore(threshold=0.5), sm.metrics.FScore(threshold=0.5)]





    print()
    print("________________________________________________________________")
    print()
    print("For NORMAL ")
    print()
    print("________________________________________________________________")
    print()
    model_arc_normal = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )
    model_arc_normal.fit(train_normal,
                validation_data=val_normal,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=VALIDATION_STEPS,
                epochs=epochs,
              )

    if(show):

        for i in range(0,5):

            show_predictions(X_test_image_normal[i:i+1] , y_test_image_normal[i],model_arc_normal)

    mean_iou_score(model_arc_normal,X_test_image_normal,y_test_image_normal,5)








    print()
    print("________________________________________________________________")
    print()
    print("For HSV ")
    print()
    print("________________________________________________________________")
    print()

    model_arc_HSV = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )

    model_arc_HSV.fit(train_HSV,
                validation_data=val_HSV,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=VALIDATION_STEPS,
                epochs=epochs,
                )

    if(show):

        for i in range(0,5):

            show_predictions(X_test_image_HSV[i:i+1] , y_test_image_HSV[i],model_arc_HSV)

    mean_iou_score(model_arc_HSV,X_test_image_HSV,y_test_image_HSV,5)






    print()
    print("________________________________________________________________")
    print()
    print("For YUV ")
    print()
    print("________________________________________________________________")
    print()


    model_arc_YUV = build_unet((256,256, 3),loss = total_loss , lr = lr , metrics = metrics )

    model_arc_YUV.fit(train_YUV,
                validation_data=val_YUV,
                steps_per_epoch=STEPS_PER_EPOCH,
                validation_steps=VALIDATION_STEPS,
                epochs=epochs,
                )

    if(show):

        for i in range(0,5):

            show_predictions(X_test_image_YUV[i:i+1] , y_test_image_YUV[i],model_arc_YUV)

    mean_iou_score(model_arc_YUV,X_test_image_YUV,y_test_image_YUV,5)






    pred1 = model_arc_normal.predict(X_test_image_normal)
    pred2 = model_arc_HSV.predict(X_test_image_HSV)
    pred3 = model_arc_YUV.predict(X_test_image_YUV)


    """
    Ensemble of the three outputs

    """

    preds=np.array([pred1, pred2, pred3])

    weights = [0.4, 0.3, 0.3]

    w = [1,1,1]
    weighted_preds = np.tensordot(preds, weights, axes=((0),(0)))
    weighted_ensemble_prediction = np.argmax(weighted_preds, axis=3)

    #(1, 256, 256, 3) (256, 256, 1) (256, 256, 1)

    print()
    print("________________________________________________________________")
    print()
    print("weighted ensemble " )
    print()
    print("________________________________________________________________")
    print()

    if(show):

        for i in range(0,5):

            display([X_test_image_normal[i:i+1] ,y_test_image_normal[i] ,  weighted_ensemble_prediction[i].reshape(256,256,1)])

    mean_iou_score_ensemble(weighted_ensemble_prediction,y_test_image_normal,n_classes = 5 )


    un_weighted_preds = np.tensordot(preds, w, axes=((0),(0)))
    un_weighted_ensemble_prediction = np.argmax(un_weighted_preds, axis=3)

    print()
    print("________________________________________________________________")
    print()
    print("un - weighted ensemble " )
    print()
    print("________________________________________________________________")
    print()

    if(show):

        for i in range(0,5):

            display([X_test_image_normal[i:i+1] ,y_test_image_normal[i] ,  un_weighted_ensemble_prediction[i].reshape(256,256,1)])

    mean_iou_score_ensemble(un_weighted_ensemble_prediction ,y_test_image_normal,n_classes = 5 )

