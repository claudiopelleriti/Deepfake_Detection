import tensorflow as tf
from enum import Enum
import sys
from keras.applications import ResNet50
from keras.applications import EfficientNetB0
from keras import layers
from keras.layers import GlobalAveragePooling2D, Dense, Attention, Dropout, BatchNormalization, Lambda
from keras.applications.efficientnet import preprocess_input
from keras.models import Model, load_model
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.utils import plot_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import os

# Funzione per preprocessare le immagini del dataset
def preprocess(image, augmentation=False):
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # Scala a [-1, 1]
    if augmentation:
        image = data_augmentation(image)
    return image
'''
def preprocess(image, augmentation=False):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.485, 0.456, 0.406])
    std = tf.constant([0.229, 0.224, 0.225])
    image = (image - mean) / std
    if augmentation:
        image = data_augmentation(image)
    return image
'''

# Definisco il layer di data augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1)
])

# Funzione per costruire il modello con o senza attention (Attention prima del pooling)
def build_model(use_attention=False, dropout_rate=0.5):
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Fine tunung: congelo tutti i layer tranne gli ultimi 10
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if use_attention:
        # Espando la dimensione con un layer Lambda
        x = Lambda(lambda t: tf.expand_dims(t, axis=1))(x)

        query = Dense(x.shape[-1])(x)
        key = Dense(x.shape[-1])(x)
        value = Dense(x.shape[-1])(x)

        x = Attention(use_scale=True)([query, key, value])

        # Rimuovo la dimensione extra con un altro Lambda
        x = Lambda(lambda t: tf.squeeze(t, axis=1))(x)

    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    output = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

'''
def build_model(use_attention=False):
    base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Fine tunung: congelo tutti i layer tranne gli ultimi 10
    for layer in base_model.layers[:-10]:
        layer.trainable = False
    for layer in base_model.layers[-10:]:
        layer.trainable = True

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if use_attention:
        x = tf.expand_dims(x, axis=1)
        query = Dense(x.shape[-1])(x)
        key = Dense(x.shape[-1])(x)
        value = Dense(x.shape[-1])(x)
        x = Attention(use_scale=True)([query, key, value])
        x = tf.squeeze(x, axis=1)

    x = Dense(128, activation='relu')(x)
    #x = Dropout(0.5)(x)
    y = Dense(2, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=y)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
'''

# Funzione per addestrare i modelli
def train_model(large_dataset, use_attention=False):
    print(f"\nInizio training: use_attention = {use_attention}")
    model_name = "model_with_attention" if use_attention else "model_no_attention"
    
    # Carico il dataset
    if large_dataset:
        dataset = tf.keras.utils.image_dataset_from_directory("dataset", image_size=(224, 224), batch_size=32, label_mode="int", shuffle=True)

        try:
            model = load_model(f"{model_name}.keras", custom_objects={"Attention": Attention, "Lambda": Lambda})
            print(f"Modello pre-addestrato {model_name} caricato.")
        
        except Exception as e:
            print(f"Errore, nessun modello pre-addestrato trovato: {e}")
            print("Addestramento da zero")
            train_model(large_dataset=False, use_attention=use_attention)
            model = load_model(f"{model_name}.keras", custom_objects={"Attention": Attention, "Lambda": Lambda})
    else:
        dataset = tf.keras.utils.image_dataset_from_directory("dataset_20k", image_size=(224, 224), batch_size=32, label_mode="int", shuffle=True)
        model = build_model(use_attention=use_attention)
    
    # Suddivido in train, val, test
    dataset_size = dataset.cardinality().numpy()
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)

    train_ds = dataset.take(train_size)
    val_ds = dataset.skip(train_size).take(val_size)
    test_ds = dataset.skip(train_size + val_size)

    # Applico il preprocessing con augmentation solo al training set
    train_ds = train_ds.map(lambda x, y: (preprocess(x, augmentation=True), y))
    val_ds = val_ds.map(lambda x, y: (preprocess(x, augmentation=False), y))

    # Ottimizzo il caricamento
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Salvo test set per usarlo successivamente
    tf.data.experimental.save(test_ds, f"{model_name}_test_set")
    print(f"Test set salvato in {model_name}_test_set")

    # EarlyStopping
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )

    # ReduceLROnPlateau
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        patience=3,
        factor=0.5,
        min_lr=1e-6
    )

    # Training
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        callbacks=[early_stopping, reduce_lr]
    )

    # Salvo il modello addestrato
    model.save(f"{model_name}.keras")
    print(f"Training completato all'epoca {len(history.history['loss'])} e modello salvato")

    # Mostro accuracy e loss
    plot_history(history, title=model_name)
    
    final_loss, final_accuracy = model.evaluate(val_ds, verbose=0)
    print(f"\nRisultati migliori del modello sul validation set:")
    print(f" - Loss: {final_loss:.4f}")
    print(f" - Accuracy: {final_accuracy:.4f}")

# Funzione per visualizzare i grafici dell'Accuracy e della Loss
def plot_history(history, title):
    plt.figure(figsize=(10, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Val')
    plt.title(f'Accuracy {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.title(f'Loss {title}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{title}_training_plot.png")
    plt.close()

# Funzione per testare il miglior modello su test_set
def evaluate_on_test_set(model_path, test_set_path):
    try:
        model = load_model(model_path, custom_objects={"Attention": Attention, "Lambda": Lambda})
    except Exception as e:
        print(f"Errore nel caricamento del modello da {model_path}: {e}")
        return
    try:
        test_ds = tf.data.experimental.load(test_set_path)
    except Exception as e:
        print(f"Errore nel caricamento del test set da {test_set_path}: {e}")
        return   
    
    test_ds = test_ds.map(lambda x, y: (preprocess(x, augmentation=False), y))
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(cm)

# Funzione di inferenza
def predict_images(model_path, image_paths):
    try:
        model = load_model(model_path, custom_objects={"Attention": Attention, "Lambda": Lambda})
    except Exception as e:
        print(f"Errore nel caricamento del modello da {model_path}: {e}")
        return [None] * len(image_paths)
    
    predictions = []
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Immagine {img_path} non trovata")
            predictions.append(None)
            continue

        try:
            img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, axis=0)
            img_array = preprocess(img_array, augmentation=False)
            pred = model.predict(img_array, verbose=0)
            predictions.append(pred)
        except Exception as e:
            print(f"Errore durante la predizione per {img_path}: {e}")
            predictions.append(None)

    return predictions

# Funzione per stampare l'help da riga di comando
def print_help():
    print("Modalità disponibili:")
    for mode in Mode:
        print(f" - {mode.value}")
    print("Se non viene specificata alcuna modalità, verrà usata 'classify' di default.")

# Funzione per impostare la modalità d'uso del programma da input in riga di comando
def get_mode_from_args(images):
    global large_dataset
    global use_attention
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if len(sys.argv) == 2:
            if arg in ("--help", "-h"):
                print_help()
                sys.exit(0)
            for mode in Mode:
                if mode == Mode.CLASSIFY:
                    if arg == mode.value:
                        print("Errore: devi inserire almeno un'immagine")
                        return "error"
                elif arg == mode.value:
                    return mode
            images.append(sys.argv[1])
            return Mode.CLASSIFY
        elif arg == Mode.TRAIN.value and ((len(sys.argv) == 3) or (len(sys.argv) == 4)):
            if len(sys.argv) == 3:
                if  sys.argv[2].lower() == 'use_large_dataset':
                    large_dataset = True
                    return Mode.TRAIN
                elif sys.argv[2].lower() == 'use_attention':
                    use_attention = True
                    return Mode.TRAIN
            else:
                if ((sys.argv[2].lower() == 'use_large_dataset') and (sys.argv[3].lower() == 'use_attention')) or ((sys.argv[2].lower() == 'use_attention') and (sys.argv[2].lower() == 'use_large_dataset')):
                    large_dataset = True
                    use_attention = True
                    return Mode.TRAIN
        else:
            index_first_image = 1
            for mode in Mode:
                if arg == mode.value:
                    if mode != Mode.CLASSIFY:
                        print(f'Errore: se vuoi utilizzare la modalità "{arg}" non devi inserire altri parametri in input dopo il nome della modalità')
                        print_help()
                        return "error"
                    else:
                        index_first_image = 2
            for i in range(index_first_image, len(sys.argv)):
                images.append(sys.argv[i])
            return Mode.CLASSIFY
    else:
        print("Errore: devi inserire il percorso di una o più immagini oppure una modalità")
        print_help()
        return "error"

# Enumeratore con le possibili funzioni del programma
class Mode(Enum):
    BUILD_MODEL = "build_model"
    TRAIN = "train"
    TEST = "test"
    CLASSIFY = "classify"


use_attention = False
large_dataset = False
# Main
if __name__ == "__main__":
    image_paths = []
    selected_mode = get_mode_from_args(image_paths)

    if selected_mode == Mode.BUILD_MODEL:
        print(f"Modalità: {selected_mode.value}")
        
        try:
            print("\nCostruzione modello SENZA attention")
            model_no_attention = build_model(use_attention=False)
            model_no_attention.summary()
            plot_model(model_no_attention, to_file="model_no_attention.png", show_shapes=True, show_layer_names=True)


            print("\nCostruzione modello CON attention")
            model_with_attention = build_model(use_attention=True)
            model_with_attention.summary()
            plot_model(model_with_attention, to_file="model_with_attention.png", show_shapes=True, show_layer_names=True)
        

        except Exception as e:
            print(f"Errore durante la costruzione dei modelli: {e}")

    elif selected_mode == Mode.TRAIN:
        print(f"Modalità: {selected_mode.value}, con dataset da {100000 if large_dataset else 20000} immagini")
        train_model(large_dataset=large_dataset, use_attention=use_attention)

    elif selected_mode == Mode.TEST:
        print(f"Modalità: {selected_mode.value}")
        evaluate_on_test_set("model_with_attention.keras", "model_with_attention_test_set")

    elif selected_mode == Mode.CLASSIFY:
        print(f"Modalità: {selected_mode.value}")
        predictions = predict_images("model_with_attention.keras", image_paths)

        class_names = ["Reale", "Generato"]
        for path, pred in zip(image_paths, predictions):
            if pred is not None:
                label = class_names[np.argmax(pred)]
                confidence = np.max(pred)
                print(f"{path}: {label} ({confidence:.2f})")
            else:
                print(f"{path}: Errore nella predizione")
    else:
        sys.exit(0)