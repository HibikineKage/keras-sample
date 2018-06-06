from keras.models import Sequential
from keras.layers import Dense

clsases=['xmu', 'others']
nb_classes = len(classes)

img_width, img_height = 150, 150

train_data_dir = 'dataset/train'
validation_data_dir = 'dataset/validation'

nb_train_samples = 200
nb_validation_samples = 50

batch_size = 10
nb_epoch = 10

train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        zoom_range=0.2,
        horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height)
        color_mode='rgb',
        classes=classes,
        class_mode='categorical'
        batch_size=batch_size,
        shuffle=True)

validation_generator =validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height)
        color_mode='rgb',
        classes=classes,
        class_mode='categorical',
        batch_size=batch_size,


