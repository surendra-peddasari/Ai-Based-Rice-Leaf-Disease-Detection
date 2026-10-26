import os
from src.data_preprocessing import get_data_generators
from src.model import build_mobilenet_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train():
    data_dir = '../data/'
    if not os.path.exists(data_dir):
        print("Data directory not found. Please populate data/ before training.")
        return

    train_gen, val_gen = get_data_generators(data_dir)
    num_classes = len(train_gen.class_indices)
    
    model = build_mobilenet_model(num_classes=num_classes)
    
    callbacks = [
        ModelCheckpoint('../outputs/best_model.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    history = model.fit(
        train_gen,
        epochs=20,
        validation_data=val_gen,
        callbacks=callbacks
    )
    print("Training complete. Best model saved to outputs/best_model.h5")

if __name__ == '__main__':
    train()
