# train_cpu.py  (save this into C:\Users\muj\colon_models)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # FORCE CPU
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.applications.inception_v3 import preprocess_input as inception_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pathlib
import random

# Optional: shap (may be slow on CPU)
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

def build_model(base_name="inception", input_shape=(299,299,3), n_classes=2, dropout=0.5):
    if base_name == "inception":
        base = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = inception_preprocess
    else:
        base = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
        preprocess = xception_preprocess

    base.trainable = False  # freeze for initial training (faster on CPU)
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(n_classes, activation='softmax')(x)
    model = models.Model(base.input, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model, preprocess, base

def plot_history(hist, out_dir, name):
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(hist.history['accuracy'], label='train acc')
    plt.plot(hist.history['val_accuracy'], label='val acc')
    plt.title(f'{name} Accuracy')
    plt.xlabel('epoch'); plt.ylabel('accuracy'); plt.legend()
    plt.subplot(1,2,2)
    plt.plot(hist.history['loss'], label='train loss')
    plt.plot(hist.history['val_loss'], label='val loss')
    plt.title(f'{name} Loss')
    plt.xlabel('epoch'); plt.ylabel('loss'); plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f'{name}_history.png')
    plt.close()

def save_confusion(y_true, y_pred, class_names, out_dir, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.ylabel('True'); plt.xlabel('Predicted')
    plt.title(f'{name} Confusion Matrix')
    plt.savefig(out_dir / f'{name}_confusion.png')
    plt.close()
    print(classification_report(y_true, y_pred, target_names=class_names))

def grad_cam(model, base_model, img, preprocess_fn, last_conv_layer_name=None):
    import numpy as np
    img_input = np.expand_dims(img, axis=0)
    x = preprocess_fn(img_input.astype(np.float32))
    if last_conv_layer_name is None:
        for layer in reversed(base_model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer_name = layer.name
                break
    grad_model = tf.keras.models.Model([model.inputs], [base_model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(x)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]
    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0,1))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1).numpy()
    cam = np.maximum(cam, 0)
    cam = cam / (cam.max() + 1e-8)
    cam = np.uint8(255 * cam)
    import cv2
    cam = cv2.resize(cam, (img.shape[1], img.shape[0]))
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    superimposed = (heatmap * 0.4 + img).astype(np.uint8)
    return superimposed, heatmap

def main(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    image_size = (299, 299)
    batch_size = args.batch_size

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, labels='inferred', label_mode='int',
        validation_split=0.30, subset='training', seed=123,
        image_size=image_size, batch_size=batch_size)

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir, labels='inferred', label_mode='int',
        validation_split=0.30, subset='validation', seed=123,
        image_size=image_size, batch_size=batch_size)

    class_names = train_ds.class_names
    n_classes = len(class_names)
    print("Classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    for base_name in ["inception", "xception"]:
        print("\n\n=== Training", base_name, "===\n")
        model, preprocess_fn, base_model = build_model(base_name=base_name, n_classes=n_classes)
        checkpoint = callbacks.ModelCheckpoint(str(out_dir / f'{base_name}_best.h5'),
                                               monitor='val_loss', save_best_only=True)
        early = callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

        history = model.fit(
    		train_ds,
    		epochs=args.epochs,
    		validation_data=val_ds,
   		callbacks=[checkpoint, early]
	)


        plot_history(history, out_dir, base_name)

        y_true = []
        y_pred = []
        for images, labels in val_ds:
            preds = model.predict(images, batch_size=batch_size)
            y_true.extend(labels.numpy().tolist())
            y_pred.extend(np.argmax(preds, axis=1).tolist())
        save_confusion(y_true, y_pred, class_names, out_dir, base_name)

        model.save(out_dir / f'{base_name}_final.h5')

    print("All done. Outputs in:", out_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    main(args)
