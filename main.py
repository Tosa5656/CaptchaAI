import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from keras import layers, ops

# --- Data Loading and Preprocessing ---
data_dir = Path("./captcha_images_v2/")
images = sorted(map(str, data_dir.glob("*.png")))
labels = [img.split(os.sep)[-1].split(".png")[0] for img in images]
characters = sorted(list(set(char for label in labels for char in label)))

print(f"Images: {len(images)}, Labels: {len(labels)}, Unique chars: {len(characters)}, Chars: {characters}")

batch_size = 16
img_width, img_height = 200, 50
max_length = max(len(label) for label in labels)
char_to_num = layers.StringLookup(vocabulary=characters, mask_token=None)
num_to_char = layers.StringLookup(vocabulary=char_to_num.get_vocabulary(), mask_token=None, invert=True)

def split_data(images, labels, train_size=0.8, shuffle=True):
    indices = keras.random.shuffle(ops.arange(len(images))) if shuffle else ops.arange(len(images))
    split_idx = int(len(images) * train_size)
    return images[indices[:split_idx]], images[indices[split_idx:]], labels[indices[:split_idx]], labels[indices[split_idx:]]

x_train, x_valid, y_train, y_valid = split_data(np.array(images), np.array(labels))

def encode_single_sample(img_path, label):
    img = tf.io.decode_png(tf.io.read_file(img_path), channels=1)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = ops.image.resize(img, [img_height, img_width])
    img = ops.transpose(img, axes=[1, 0, 2])
    label = char_to_num(tf.strings.unicode_split(label, "UTF-8"))
    return {"image": img, "label": label}

train_dataset = (tf.data.Dataset.from_tensor_slices((x_train, y_train))
    .map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE))

validation_dataset = (tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
    .map(encode_single_sample, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(buffer_size=tf.data.AUTOTUNE))

# --- Visualization ---
_, ax = plt.subplots(4, 4, figsize=(10, 5))
for batch in train_dataset.take(1):
    for i in range(16):
        img = (batch["image"][i] * 255).numpy().astype("uint8")
        label = tf.strings.reduce_join(num_to_char(batch["label"][i])).numpy().decode("utf-8")
        ax[i // 4, i % 4].imshow(img[:, :, 0].T, cmap="gray")
        ax[i // 4, i % 4].set_title(label)
        ax[i // 4, i % 4].axis("off")
plt.show()

# --- CTC Loss and Decoding ---
def ctc_batch_cost(y_true, y_pred, input_length, label_length):
    label_length = ops.cast(ops.squeeze(label_length, -1), "int32")
    input_length = ops.cast(ops.squeeze(input_length, -1), "int32")
    sparse_labels = ops.cast(ctc_label_dense_to_sparse(y_true, label_length), "int32")
    y_pred = ops.log(ops.transpose(y_pred, [1, 0, 2]) + keras.backend.epsilon())
    return ops.expand_dims(tf.compat.v1.nn.ctc_loss(inputs=y_pred, labels=sparse_labels, sequence_length=input_length), 1)

def ctc_label_dense_to_sparse(labels, label_lengths):
    label_shape = ops.shape(labels)
    num_batches_tns = ops.stack([label_shape[0]])
    max_num_labels_tns = ops.stack([label_shape[1]])
    
    def range_less_than(old_input, current_input):
        return ops.expand_dims(ops.arange(ops.shape(old_input)[1]), 0) < tf.fill(max_num_labels_tns, current_input)

    init = ops.cast(tf.fill([1, label_shape[1]], 0), dtype="bool")
    dense_mask = tf.compat.v1.scan(range_less_than, label_lengths, initializer=init, parallel_iterations=1)[:, 0, :]

    label_array = ops.reshape(ops.tile(ops.arange(0, label_shape[1]), num_batches_tns), label_shape)
    label_ind = tf.compat.v1.boolean_mask(label_array, dense_mask)
    
    batch_array = ops.transpose(ops.reshape(ops.tile(ops.arange(0, label_shape[0]), max_num_labels_tns), tf.reverse(label_shape, [0])))
    batch_ind = tf.compat.v1.boolean_mask(batch_array, dense_mask)
    indices = ops.transpose(ops.reshape(ops.concatenate([batch_ind, label_ind], axis=0), [2, -1]))
    vals_sparse = tf.compat.v1.gather_nd(labels, indices)
    return tf.SparseTensor(ops.cast(indices, "int64"), vals_sparse, ops.cast(label_shape, "int64"))

class CTCLayer(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.loss_fn = ctc_batch_cost
    def call(self, y_true, y_pred):
        batch_len = ops.cast(ops.shape(y_true)[0], "int64")
        input_length = ops.cast(ops.shape(y_pred)[1], "int64")
        label_length = ops.cast(ops.shape(y_true)[1], "int64")
        input_length = input_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * ops.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)
        return y_pred
    def get_config(self):
       config = super().get_config()
       return config
   
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def ctc_decode(y_pred, input_length, greedy=True, beam_width=100, top_paths=1):
    input_shape = ops.shape(y_pred)
    num_samples, num_steps = input_shape[0], input_shape[1]
    y_pred = ops.log(ops.transpose(y_pred, axes=[1, 0, 2]) + keras.backend.epsilon())
    input_length = ops.cast(input_length, dtype="int32")
    if greedy:
        (decoded, log_prob) = tf.nn.ctc_greedy_decoder(inputs=y_pred, sequence_length=input_length)
    else:
        (decoded, log_prob) = tf.compat.v1.nn.ctc_beam_search_decoder(
            inputs=y_pred, sequence_length=input_length, beam_width=beam_width, top_paths=top_paths
        )
    decoded_dense = []
    for st in decoded:
        st = tf.SparseTensor(st.indices, st.values, (num_samples, num_steps))
        decoded_dense.append(tf.sparse.to_dense(sp_input=st, default_value=-1))
    return (decoded_dense, log_prob)

def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    results = ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :max_length]
    output_text = [tf.strings.reduce_join(num_to_char(res)).numpy().decode("utf-8") for res in results]
    return output_text

# --- Model Building ---
def build_model():
    input_img = layers.Input(shape=(img_width, img_height, 1), name="image", dtype="float32")
    labels = layers.Input(name="label", shape=(None,), dtype="float32")

    x = layers.Conv2D(32, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv1")(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", kernel_initializer="he_normal", padding="same", name="Conv2")(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    new_shape = ((img_width // 4), (img_height // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="relu", name="dense1")(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True, dropout=0.25))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True, dropout=0.25))(x)
    x = layers.Dense(len(char_to_num.get_vocabulary()) + 1, activation="softmax", name="dense2")(x)
    
    output = CTCLayer(name="ctc_loss")(labels, x)
    model = keras.models.Model(inputs=[input_img, labels], outputs=output, name="ocr_model_v1")
    model.compile(optimizer=keras.optimizers.Adam())
    return model

# --- Training or Loading and Prediction ---
model_file = 'model.keras'
if os.path.exists(model_file):
    print(f"Model file '{model_file}' found. Loading model...")
    model = keras.models.load_model(model_file, custom_objects={'CTCLayer': CTCLayer})
    print("Model loaded successfully.")
    
    prediction_model = keras.models.Model(model.input[0], model.get_layer(name="dense2").output)
    prediction_model.summary()
    
    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]
        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        orig_texts = [tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8") for label in batch_labels]
        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8).T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
        plt.show()
else:
    print(f"Model file '{model_file}' not found. Training new model...")
    model = build_model()
    model.summary()

    epochs = 200
    early_stopping_patience = 10
    early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=early_stopping_patience, restore_best_weights=True)

    history = model.fit(train_dataset, validation_data=validation_dataset, epochs=epochs, callbacks=[early_stopping])

    model.save(model_file)
    print(f"Model saved to '{model_file}'.")

    prediction_model = keras.models.Model(model.input[0], model.get_layer(name="dense2").output)
    prediction_model.summary()

    for batch in validation_dataset.take(1):
        batch_images = batch["image"]
        batch_labels = batch["label"]
        preds = prediction_model.predict(batch_images)
        pred_texts = decode_batch_predictions(preds)
        orig_texts = [tf.strings.reduce_join(num_to_char(label)).numpy().decode("utf-8") for label in batch_labels]
        _, ax = plt.subplots(4, 4, figsize=(15, 5))
        for i in range(len(pred_texts)):
            img = (batch_images[i, :, :, 0] * 255).numpy().astype(np.uint8).T
            title = f"Prediction: {pred_texts[i]}"
            ax[i // 4, i % 4].imshow(img, cmap="gray")
            ax[i // 4, i % 4].set_title(title)
            ax[i // 4, i % 4].axis("off")
        plt.show()