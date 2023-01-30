



def process_x(files,dsize = (256,256) , con = "Normal"):

    """
    files: arrays of actual image paths
    """

    print("creating  images-- to arrays ",con)

    x = []

    for f in files:

        img =  cv2.imread(f,1)

        if con == "YUV":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

        elif con == "HSV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        elif con == "LUV":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)

        elif con == "LAB":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

        elif con == "HLS":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

        elif con == "GRAY":

            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#LUV LAB HLS GRAY

        img = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)/255.0
        x.append(img)
    return np.array(x).astype("float32")

def process_y(files,dsize = (256,256),cat = False , n_classes = 5 ):
    print("creating  images-- to arrays")
    masks = []
    for f in files:
        img  = cv2.imread(f,0)
        res = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)
        masks.append(res)
    masks = np.array(masks)
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    train_masks_reshaped = masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
    if(cat):
        train_masks_cat = to_categorical(train_masks_input, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((train_masks_input.shape[0], train_masks_input.shape[1], train_masks_input.shape[2], n_classes))
        return y_train_cat
    else:
        return train_masks_input


def process_x1(files,dsize = (256,256) , con = "Normal" ):

  # files = [i for i in files]
  print("creating  images-- to arrays" , con )
  x = []
  X = []
  for f in files:
    #   img =  cv2.imread(f,1)
    #   if con == "YUV":
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    #   elif con == "HSV":
    #     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    #   img = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)/255.0
    #   x.append(img)
    img =  cv2.imread(f,1)
    if con == "YUV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    elif con == "HSV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif con == "LUV":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LUV)
    elif con == "LAB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    elif con == "HLS":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    elif con == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#LUV LAB HLS GRAY

    img = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)/255.0
    x.append(img)
  for i in x :
      image = tf.cast(tf.convert_to_tensor(i), tf.float32)
      X.append(image)
  return X

def process_y1(files,dsize = (256,256) , cat = False , n_classes = 5 ):
    print("creating  images-- to arrays")
    masks = []
    Y = []
    for f in files:
        img  = cv2.imread(f,0)
        res = cv2.resize(img, dsize= dsize, interpolation = cv2.INTER_NEAREST)
        masks.append(tf.convert_to_tensor(res))
    masks = np.array(masks)
    labelencoder = LabelEncoder()
    n, h, w = masks.shape
    train_masks_reshaped = masks.reshape(-1,1)
    train_masks_reshaped_encoded = labelencoder.fit_transform(train_masks_reshaped)
    train_masks_encoded_original_shape = train_masks_reshaped_encoded.reshape(n, h, w)
    if(cat == False):

        # train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
        for mask in  train_masks_encoded_original_shape:
            m = tf.expand_dims(tf.convert_to_tensor(mask), axis=-1)
            m = tf.cast(m, tf.float32)
            Y.append(m)
        return Y
    else:
        train_masks_input = np.expand_dims(train_masks_encoded_original_shape, axis=3)
        train_masks_cat = to_categorical(train_masks_input, num_classes=n_classes)
        y_train_cat = train_masks_cat.reshape((train_masks_input.shape[0], train_masks_input.shape[1], train_masks_input.shape[2], n_classes))
        for mask in y_train_cat:
             m = tf.cast(tf.convert_to_tensor(mask), tf.float32)
             Y.append(m)
        return Y