from my_utils import split_data, order_test_set, create_generators
from keras.callbacks import ModelCheckpoint, EarlyStopping
from deeplearning_models import streetsigns_model
import tensorflow as tf

if __name__=="__main__":
    SPLIT = False
    TRAIN = False
    TEST = True

    # STEP 1: Split into Triain + Validation Data
    path_to_data = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\Train" # path to base dataset
    path_to_train = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\training_data\\train" # path to new training-folder
    path_to_val = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\training_data\\val" # path to new val-folder
    path_to_test = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\Test"

    path_to_images = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\Test"
    path_to_csv = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\Test.csv"
    order_test_set(path_to_images, path_to_csv)



    if SPLIT:
        split_data(path_to_data, path_to_train=path_to_train, path_to_val=path_to_val)




    # STEP 2: Create a Generator

    batch_size = 64
    epochs = 15

    train_generator, val_generator, test_generator = create_generators(batch_size, path_to_train, path_to_val, path_to_test)

    # STEP 3: Pass Generator to DL Model

    nr_classes = train_generator.num_classes
    model = streetsigns_model(nr_classes)


    if TRAIN:


        # STEP 3.2: Create a callback(S) to only keep the best model

        path_to_saved_model = "./Models"    # Save callbacks to /Models

        cp_saver = ModelCheckpoint(
            path_to_saved_model,
            monitor = 'val_accuracy',       # Monitor the validation accuracy to find the best model (could also use val_loss, but then use mode = 'min')
            mode = 'max',                   # Save models with higher val_accuracy than previous model
            save_best_only = True,          # Save only the best model
            save_freq = 'epoch',
            verbose = 1
        )
        

        early_stop = EarlyStopping(
            monitor = 'val_accuracy',   
            patience = 10,             # Stop if after 10 epochs the val_accuracy does not go higher
        )


        # STEP 4: Compiling (Optimizing)
        model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

        # STEP 5: Model Fitting
        model.fit(train_generator,
                epochs = epochs,
                batch_size = batch_size,
                validation_data = val_generator,
                callbacks = [cp_saver, early_stop] # Can use multiple callbacks...
                )




    # STEP 6: Use Saved Model for testing

    if TEST:
        model = tf.keras.models.load_model('./Models')
        model.summary() # Print model's architecture

        print("Evaluating validation set:")
        model.evaluate(val_generator)
        print("Evaluating test set:")
        model.evaluate(test_generator) # You want test_accuracy =~ val_accuracy




