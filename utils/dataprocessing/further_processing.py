import tensorflow as tf


def brightness(img, mask):
    img = tf.image.adjust_brightness(img, 0.1)
    return img, mask

def gamma(img, mask):
    img = tf.image.adjust_gamma(img, 0.1)
    return img, mask

def hue(img, mask):
    img = tf.image.adjust_hue(img, -0.1)
    return img, mask


def flip_hori(img, mask):
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask

def flip_vert(img, mask):
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask

def rotate(img, mask):
    img = tf.image.rot90(img)
    mask = tf.image.rot90(mask)
    return img, mask

def further_process(X_train,y_train,X_val,y_val,BATCH = 4 ):

    train_X = tf.data.Dataset.from_tensor_slices(X_train)
    val_X = tf.data.Dataset.from_tensor_slices(X_val)

    train_y = tf.data.Dataset.from_tensor_slices(y_train)
    val_y = tf.data.Dataset.from_tensor_slices(y_val)



    train = tf.data.Dataset.zip((train_X, train_y))
    val = tf.data.Dataset.zip((val_X, val_y))


    a = train.map(brightness)
    b = train.map(gamma)
    c = train.map(hue)
    d = train.map(flip_hori)
    e = train.map(flip_vert)
    f = train.map(rotate)

    train = train.concatenate(a)
    train = train.concatenate(b)
    train = train.concatenate(c)
    train = train.concatenate(d)
    train = train.concatenate(e)
    train = train.concatenate(f)


    AT = tf.data.AUTOTUNE
    BUFFER = len(X_train) + len(X_val)


    train = train.cache().shuffle(BUFFER).batch(BATCH).repeat()
    train = train.prefetch(buffer_size=AT)
    val = val.batch(BATCH)

    return (train,val)