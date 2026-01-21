"""
AI vs Real Image Classifier - Training Module
This module handles data preparation, augmentation, and model training.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import shutil

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001


class DataPreparation:
    """Handle data organization and splitting"""

    def __init__(self, dataset_path='dataset'):
        self.dataset_path = dataset_path
        self.train_dir = 'data_split/train'
        self.val_dir = 'data_split/val'
        self.test_dir = 'data_split/test'

    def split_data(self):
        """Split data into train/val/test sets maintaining class distribution"""
        print("Starting data split...")

        # Create directories
        for directory in [self.train_dir, self.val_dir, self.test_dir]:
            os.makedirs(os.path.join(directory, 'real'), exist_ok=True)
            os.makedirs(os.path.join(directory, 'ai_generated'), exist_ok=True)

        # Process each class
        for class_name in ['real', 'ai_generated']:
            class_path = os.path.join(self.dataset_path, class_name)
            images = [f for f in os.listdir(class_path)
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]

            print(f"\n{class_name}: {len(images)} images")

            # Split: 70% train, 15% val, 15% test
            train_imgs, temp_imgs = train_test_split(images, test_size=0.3, random_state=42)
            val_imgs, test_imgs = train_test_split(temp_imgs, test_size=0.5, random_state=42)

            print(f"  Train: {len(train_imgs)}, Val: {len(val_imgs)}, Test: {len(test_imgs)}")

            # Copy files
            for img in train_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(self.train_dir, class_name, img)
                )

            for img in val_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(self.val_dir, class_name, img)
                )

            for img in test_imgs:
                shutil.copy2(
                    os.path.join(class_path, img),
                    os.path.join(self.test_dir, class_name, img)
                )

        print("\nData split completed!")
        return self.train_dir, self.val_dir, self.test_dir


class ModelBuilder:
    """Build and compile the classification model"""

    def __init__(self, img_size=IMG_SIZE):
        self.img_size = img_size

    def build_model(self):
        """Create model with transfer learning and custom layers"""
        print("\nBuilding model...")

        # Clear session
        keras.backend.clear_session()

        # Build the model using Functional API
        inputs = keras.Input(shape=(self.img_size, self.img_size, 3))

        # Data augmentation (only during training)
        x = layers.RandomFlip("horizontal")(inputs)
        x = layers.RandomRotation(0.2)(x)
        x = layers.RandomZoom(0.2)(x)
        x = layers.RandomTranslation(0.1, 0.1)(x)
        x = layers.RandomContrast(0.2)(x)

        # Preprocessing for MobileNetV2
        x = keras.applications.mobilenet_v2.preprocess_input(x)

        # Create base model - MobileNetV2 is more stable
        print("Loading MobileNetV2 with ImageNet weights...")
        base_model = MobileNetV2(
            include_top=False,
            weights='imagenet',
            input_shape=(self.img_size, self.img_size, 3),
            pooling='avg'
        )

        print("MobileNetV2 loaded successfully!")

        # Freeze base model initially
        base_model.trainable = False

        # Apply base model
        x = base_model(x, training=False)

        # Custom classification head
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001))(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=inputs, outputs=outputs)

        # Compile model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )

        print(f"Model built successfully!")
        print(f"Total parameters: {model.count_params():,}")

        return model, base_model


class ModelTrainer:
    """Handle model training with callbacks and fine-tuning"""

    def __init__(self, model, base_model, train_dir, val_dir, test_dir):
        self.model = model
        self.base_model = base_model
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.history = None

    def prepare_data_generators(self):
        """Create data generators with preprocessing"""
        # Simple rescaling - preprocessing is done in the model
        train_datagen = ImageDataGenerator(rescale=1./255)
        val_test_datagen = ImageDataGenerator(rescale=1./255)

        train_gen = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=True
        )

        val_gen = val_test_datagen.flow_from_directory(
            self.val_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

        test_gen = val_test_datagen.flow_from_directory(
            self.test_dir,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode='binary',
            shuffle=False
        )

        # Save class indices
        with open('class_indices.json', 'w') as f:
            json.dump(train_gen.class_indices, f)

        print(f"\nClass mapping: {train_gen.class_indices}")

        return train_gen, val_gen, test_gen

    def calculate_class_weights(self, train_gen):
        """Calculate class weights to handle imbalance"""
        # Count samples per class
        class_counts = {}
        for class_name in train_gen.class_indices:
            class_path = os.path.join(self.train_dir, class_name)
            class_counts[train_gen.class_indices[class_name]] = len(os.listdir(class_path))

        total = sum(class_counts.values())
        class_weights = {k: total / (len(class_counts) * v) for k, v in class_counts.items()}

        print(f"\nClass weights: {class_weights}")
        return class_weights

    def train_initial(self, train_gen, val_gen, class_weights):
        """Initial training with frozen base model"""
        print("\n" + "=" * 50)
        print("PHASE 1: Initial Training (Base Frozen)")
        print("=" * 50)

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'model_phase1.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )
        ]

        history1 = self.model.fit(
            train_gen,
            epochs=15,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return history1

    def fine_tune(self, train_gen, val_gen, class_weights):
        """Fine-tune top layers of base model"""
        print("\n" + "=" * 50)
        print("PHASE 2: Fine-Tuning")
        print("=" * 50)

        # Unfreeze top layers of base model
        self.base_model.trainable = True

        # Freeze all layers except the last 30
        for layer in self.base_model.layers[:-30]:
            layer.trainable = False

        print(f"Trainable layers: {sum([1 for l in self.model.layers if l.trainable])}")

        # Recompile with lower learning rate
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE / 10),
            loss='binary_crossentropy',
            metrics=['accuracy', keras.metrics.AUC(name='auc'),
                     keras.metrics.Precision(name='precision'),
                     keras.metrics.Recall(name='recall')]
        )

        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=8,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                'best_model.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=4,
                min_lr=1e-8,
                verbose=1
            )
        ]

        history2 = self.model.fit(
            train_gen,
            epochs=EPOCHS,
            validation_data=val_gen,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )

        return history2

    def evaluate_model(self, test_gen):
        """Evaluate model on test set"""
        print("\n" + "=" * 50)
        print("MODEL EVALUATION ON TEST SET")
        print("=" * 50)

        # Load best model with custom objects
        print("Loading best model...")
        best_model = keras.models.load_model('best_model.keras')

        # Evaluate
        test_loss, test_acc, test_auc, test_prec, test_recall = best_model.evaluate(test_gen, verbose=1)

        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test AUC: {test_auc:.4f}")
        print(f"Test Precision: {test_prec:.4f}")
        print(f"Test Recall: {test_recall:.4f}")

        # Predictions for confusion matrix
        test_gen.reset()
        predictions = best_model.predict(test_gen, verbose=1)
        y_pred = (predictions > 0.5).astype(int).flatten()
        y_true = test_gen.classes

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['AI Generated', 'Real'],
                    yticklabels=['AI Generated', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("\nConfusion matrix saved as 'confusion_matrix.png'")

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred,
                                    target_names=['AI Generated', 'Real']))

        # Save metrics
        metrics = {
            'test_accuracy': float(test_acc),
            'test_auc': float(test_auc),
            'test_precision': float(test_prec),
            'test_recall': float(test_recall),
            'test_loss': float(test_loss)
        }

        with open('test_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

        return metrics

    def plot_training_history(self, history1, history2):
        """Plot training and validation metrics"""
        # Combine histories
        metrics_to_plot = ['accuracy', 'loss', 'auc']

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]

            # Combine phase 1 and phase 2
            train_values = history1.history[metric] + history2.history[metric]
            val_values = history1.history[f'val_{metric}'] + history2.history[f'val_{metric}']

            epochs_range = range(1, len(train_values) + 1)
            phase1_end = len(history1.history[metric])

            ax.plot(epochs_range, train_values, label=f'Training {metric.capitalize()}', linewidth=2)
            ax.plot(epochs_range, val_values, label=f'Validation {metric.capitalize()}', linewidth=2)
            ax.axvline(x=phase1_end, color='red', linestyle='--', label='Fine-tuning Start')
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(metric.capitalize(), fontsize=12)
            ax.set_title(f'{metric.capitalize()} Over Time', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # Remove extra subplot
        fig.delaxes(axes[3])

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        print("\nTraining history saved as 'training_history.png'")
        plt.close()


def main():
    """Main training pipeline"""
    print("=" * 60)
    print("AI vs Real Image Classifier - Training Pipeline")
    print("=" * 60)

    # Step 1: Prepare data
    data_prep = DataPreparation()
    train_dir, val_dir, test_dir = data_prep.split_data()

    # Step 2: Build model
    model_builder = ModelBuilder()
    model, base_model = model_builder.build_model()

    # Step 3: Train model
    trainer = ModelTrainer(model, base_model, train_dir, val_dir, test_dir)
    train_gen, val_gen, test_gen = trainer.prepare_data_generators()

    # Calculate class weights
    class_weights = trainer.calculate_class_weights(train_gen)

    # Phase 1: Initial training
    history1 = trainer.train_initial(train_gen, val_gen, class_weights)

    # Phase 2: Fine-tuning
    history2 = trainer.fine_tune(train_gen, val_gen, class_weights)

    # Step 4: Evaluate
    metrics = trainer.evaluate_model(test_gen)

    # Step 5: Plot history
    trainer.plot_training_history(history1, history2)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print(f"\nFinal Test Accuracy: {metrics['test_accuracy']:.4f}")
    print("\nGenerated files:")
    print("  - best_model.keras (trained model)")
    print("  - class_indices.json (class mapping)")
    print("  - test_metrics.json (evaluation metrics)")
    print("  - confusion_matrix.png (confusion matrix plot)")
    print("  - training_history.png (training curves)")
    print("\nYou can now use predict.py for predictions and app.py for the web interface!")


if __name__ == "__main__":
    main()