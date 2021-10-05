# Disable info log messages to get a cleaner terminal
# Can only be done before importing tf
import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
logging.getLogger('tensorflow').setLevel(logging.WARN)

from tensorflow import keras
from tensorflow.keras.applications import vgg19
import numpy as np
import tensorflow as tf
import time


print('Tensorflow version:', tf.__version__)
print('Available GPU:', tf.test.gpu_device_name())

# For some reason this is needed to avoid problems in some cases maybe?
gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)


class StyleTransferConfig:
    def __init__(self,
        num_iterations = 5000,
        save_interval = 100,
        total_variation_weight = 1e-6,
        style_weight = 1e-6,
        content_weight = 2.5e-5,
        content_layer_name = "block5_conv2",
        style_layers_names = [
            'block1_conv1',
            'block2_conv1',
            'block3_conv1',
            'block4_conv1',
            'block5_conv1'],
        start_from_content = True, # Start from content img instead of noise
        optimizer = keras.optimizers.SGD(
            keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=100.0,
                decay_steps=100,
                decay_rate=0.96  # 0.96
            )
        )
    ):
        self.num_iterations = num_iterations
        self.save_interval = save_interval
        self.total_variation_weight = total_variation_weight
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.content_layer_name = content_layer_name
        self.style_layers_names = style_layers_names
        self.start_from_content = start_from_content
        self.optimizer = optimizer

    def __str__(self):
        string = ""
        for attr, value in vars(self).items():
            string += f"{attr}: {value}\n"
        string += f"\tlearning rate: {self.optimizer.learning_rate}\n"
        for attr, value in vars(self.optimizer.learning_rate).items():
            string += f"\t\t{attr}: {value}\n"
        return string


def neural_style_transfer(content_img_url: str, style_img_url:str, config: StyleTransferConfig):
    content_img_path = keras.utils.get_file(os.path.basename(content_img_url), content_img_url)
    style_img_path = keras.utils.get_file(os.path.basename(style_img_url), style_img_url)


    content_width, content_height = keras.preprocessing.image.load_img(
        content_img_path).size
    n_rows = 400
    n_cols = int(content_width * n_rows / content_height)


    def preprocess_img(img_path):
        img = keras.preprocessing.image.load_img(
            img_path, target_size=(n_rows, n_cols))
        img = keras.preprocessing.image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = vgg19.preprocess_input(img)
        return tf.convert_to_tensor(img)


    def deprocess_img(x):
        x = x.reshape((n_rows, n_cols, 3))
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = x[:, :, ::-1]  # BGR -> RGB
        x = np.clip(x, 0, 255).astype("uint8")
        return x


    def content_loss(content, combination):
        return tf.reduce_sum(tf.square(combination - content))


    def total_variation_loss(x):
        a = tf.square(
            x[:, : n_rows - 1, : n_cols - 1, :] - x[:, 1:, : n_cols - 1, :]
        )
        b = tf.square(
            x[:, : n_rows - 1, : n_cols - 1, :] - x[:, : n_rows - 1, 1:, :]
        )
        return tf.reduce_sum(tf.pow(a + b, 1.25))


    def gram_matrix(x):
        x = tf.transpose(x, (2, 0, 1))
        features = tf.reshape(x, (tf.shape(x)[0], -1))
        gram = tf.matmul(features, tf.transpose(features))
        return gram


    # The "style loss" is designed to maintain
    # the style of the reference image in the generated image.
    # It is based on the gram matrices (which capture style) of
    # feature maps from the style reference image
    # and from the generated image
    def style_loss(style, combination):
        S = gram_matrix(style)
        C = gram_matrix(combination)
        channels = 3
        size = n_rows * n_cols
        return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels ** 2) * (size ** 2))


    # Using pretrained weights from imagenet
    model = vgg19.VGG19(weights="imagenet", include_top=False)
    outputs = dict([(layer.name, layer.output) for layer in model.layers])
    feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs)


    def compute_loss(combination_img, content_img, style_img):
        # Concatenate tensors along dimension 0
        input_tensor = tf.concat([content_img, style_img, combination_img], axis=0)
        features = feature_extractor(input_tensor)

        loss = tf.zeros(shape=())  # WHY shape=()

        # Style loss
        for layer_name in config.style_layers_names:
            layer_features = features[layer_name]
            style_reference_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_reference_features, combination_features)
            loss += (config.style_weight / len(config.style_layers_names)) * sl

        # Total variation loss
        loss += config.total_variation_weight * total_variation_loss(combination_img)

        # Content loss
        layer_features = features[config.content_layer_name]
        content_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]
        loss = loss + config.content_weight * content_loss(content_features, combination_features)

        return loss


    @tf.function
    def compute_loss_and_gradients(combination_img, content_img, style_img):
        with tf.GradientTape() as tape:
            loss = compute_loss(combination_img, content_img, style_img)
        gradients = tape.gradient(loss, combination_img)
        return loss, gradients


    content_img = preprocess_img(content_img_path)
    style_img = preprocess_img(style_img_path)

    if config.start_from_content:
        combination_img = tf.Variable(preprocess_img(content_img_path))
    else:
        combination_img = tf.Variable(tf.random.uniform(
            content_img.shape, minval=-128, maxval=128  # No idea what im doing
        ))

    time_str = time.strftime("%Y%m%d-%H%M%S")
    currentDir = os.getcwd()
    save_directory = f"{currentDir}/output/started-at-{time_str}"
    os.makedirs(save_directory)

    keras.preprocessing.image.save_img(
        save_directory + "/result-at-0-iterations.png", deprocess_img(combination_img.numpy()))
    keras.preprocessing.image.save_img(
        save_directory + "/content.png", deprocess_img(content_img.numpy()))
    keras.preprocessing.image.save_img(
        save_directory + "/style.png", deprocess_img(style_img.numpy())) # TODO Fix stretch

    with open(save_directory + '/config.txt', 'a') as the_file:
        the_file.write(str(config))

    iterations_info = ""

    start_time = time.perf_counter()
    for i in range(1, config.num_iterations + 1):
        loss, gradients = compute_loss_and_gradients(
            combination_img, content_img, style_img)

        config.optimizer.apply_gradients([(gradients, combination_img)])

        if i % config.save_interval == 0 or i == config.num_iterations:
            elapsed_time = (time.perf_counter() - start_time) / 60
            iteration_info = f"Iteration {i}, loss = {loss:.2f}, elapsed time = {elapsed_time:.2f} min"
            print(iteration_info)
            iterations_info += iteration_info + '\n'
            img = deprocess_img(combination_img.numpy())

            file_name = f"result-at-{i}-iterations.png"
            keras.preprocessing.image.save_img(f"{save_directory}/{file_name}", img)

    with open(save_directory + '/iterations.txt', 'a') as the_file:
        the_file.write(iterations_info)
