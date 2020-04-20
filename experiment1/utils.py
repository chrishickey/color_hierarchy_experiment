import cv2, os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def opp_transform(image):
    # split the image into its respective RGB components
    (B, G, R) = cv2.split(image.astype("float"))
    # compute rg = R - G
    O1 = ((R + G + B) -1.5)/ 1.5
    O2 = ((R - G))
    O3 = ((R + G) - (2 * B))/2
    image = cv2.merge((O1,O2,O3))
    return image


def hsv_transform(image):
    image = np.array(image)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
    return hsv_image

def bgr_transform(image):
    image = np.array(image)
    bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return bgr_image

def lab_transform(image):
    image = np.array(image)
    lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2Lab)
    return lab_image

def xyz_transform(image):
    image = np.array(image)
    xyz_image = cv2.cvtColor(image, cv2.COLOR_RGB2XYZ)
    return xyz_image

def yCrCb_transform(image):
    image = np.array(image)
    yrb_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return yrb_image

def luv_transform(image):
    image = np.array(image)
    luv_image = cv2.cvtColor(image, cv2.COLOR_RGB2Luv)
    return luv_image

def yuv_transform(image):
    image = np.array(image)
    yuv_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    return yuv_image

def _get_preprocessing_function(transform):
    if transform and transform.lower() == 'opp':
        print('Using opp')
        preprocessing_function = opp_transform
    elif transform and transform.lower() == 'hsv':
        preprocessing_function = hsv_transform
    elif transform and transform.lower() == 'bgr':
        print('Using bgr')
        preprocessing_function = bgr_transform
    elif transform and transform.lower() == 'lab':
        preprocessing_function = lab_transform
    elif transform and transform.lower() == 'xyz':
        preprocessing_function = xyz_transform
    elif transform and transform.lower() == 'ybr':
        print('Using yCrCb')
        preprocessing_function = yCrCb_transform
    elif transform and transform.lower() == 'luv':
        preprocessing_function = luv_transform
    elif transform and transform.lower() == 'yuv':
        print('Using yuv')
        preprocessing_function = yuv_transform
    else:
        print('Using rgb')
        preprocessing_function = None

    return preprocessing_function


def _get_num_files(dr):
    return sum([len(files) for r, d, files in os.walk(dr)])


def get_training_data(train_dir, colours, img_rows, img_cols, batch_size=16, transform='rgb', rescale=False):
    preprocessing_function = _get_preprocessing_function(transform)

    if rescale:
        data_generator = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            preprocessing_function=preprocessing_function,
            rescale=1. / 255,
        )
    else:
        data_generator = ImageDataGenerator(
            shear_range=0.2,
            zoom_range=0.3,
            horizontal_flip=True,
            preprocessing_function=preprocessing_function,
        )

    return data_generator.flow_from_directory(
        train_dir,
        shuffle=True,
        classes=colours,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical')


def get_test_data(test_data, colours, img_rows, img_cols, transform='rgb', rescale=False):
    preprocessing_function = _get_preprocessing_function(transform)

    if rescale:
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rescale=1. / 255,
        )
    else:
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
        )

    return data_generator.flow_from_directory(
        test_data,
        classes=colours,
        target_size=(img_rows, img_cols),
        batch_size=_get_num_files(test_data),
        class_mode='categorical')


def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        os.mkdir(string)
        return string
