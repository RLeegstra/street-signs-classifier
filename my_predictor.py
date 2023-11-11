import tensorflow as tf
import numpy as np



# Predict street sign speed using our model


def predict_with_model(model, img_path):
    image = tf.io.read_file(img_path)
    image = tf.image.decode_png(image, channels = 3) # all data images are png. 3 Channels because R.G.B.
    image = tf.image.convert_image_dtype(image, dtype = tf.float32) # convert pixels from unsigned int, to float32 (scale from 0 to 1 instead of 0 to 255)
    image = tf.image.resize(image, [60, 60]) # resize image to 60x60 (remember model input)
    image = tf.expand_dims(image, axis = 0) # add dimension from (60,60,3) -> (1,60,60,3) (required for model)

    predictions = model.predict(image) # returns a list of label probabilities (for example [0.001, 0.00003, ...., 0.9, 0.003])
    probability = np.max(predictions)

    predictions = np.argmax(predictions)


    return predictions, probability









if __name__=="__main__":

    img_path = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\Test\\2\\00409.png"
    img_path = "C:\\Users\\Ruben\\Desktop\\Datasets\\street_signs\\Test\\0\\00807.png"

    model = tf.keras.models.load_model('./Models')


    prediction, prediction_probability = predict_with_model(model, img_path)
    print(f"prediction = {prediction}")
    print(f"likelihood = {prediction_probability}")