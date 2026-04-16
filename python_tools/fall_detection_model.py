#!/usr/bin/env python3
"""
fall_detection_model.py - Train Fall Detection Model
=====================================================
Trains a neural network to classify fall vs normal activities
using accelerometer and gyroscope data from MPU6050.

Output: model.h for Arduino/ESP8266
"""

import numpy as np
import os

# ── Configuration ─────────────────────────────────────────────
WINDOW_SIZE = 25
SAMPLE_RATE = 50  # Hz
NUM_FEATURES = 64
NUM_CHANNELS = 8  # ax, ay, az, gx, gy, gz, accel_mag, gyro_mag
FEATURES_PER_CHANNEL = 8  # mean, std, energy, max, min, peak_idx, rms, range

# ── Synthetic Data Generation ───────────────────────────────────
def generate_synthetic_data(n_samples=1000):
    """Generate synthetic sensor data for training."""
    np.random.seed(42)
    
    X, y = [], []
    
    for _ in range(n_samples // 2):
        # Normal activities (walking, standing, sitting)
        window = generate_normal_activity(duration=WINDOW_SIZE)
        X.append(window)
        y.append(0)
        
    for _ in range(n_samples // 2):
        # Fall events
        window = generate_fall_event(duration=WINDOW_SIZE)
        X.append(window)
        y.append(1)
    
    return np.array(X), np.array(y)

def generate_normal_activity(duration=25):
    """Generate normal activity sensor pattern."""
    data = {
        'ax': np.random.randn(duration) * 0.1 + np.sin(np.arange(duration) * 0.3) * 0.5,
        'ay': np.random.randn(duration) * 0.1 + 1.0,
        'az': np.random.randn(duration) * 0.1 + 0.2,
        'gx': np.random.randn(duration) * 5,
        'gy': np.random.randn(duration) * 5,
        'gz': np.random.randn(duration) * 5,
    }
    return compute_channels(data)

def generate_fall_event(duration=25):
    """Generate fall event sensor pattern (free-fall + impact)."""
    data = {
        'ax': np.random.randn(duration) * 0.3,
        'ay': np.random.randn(duration) * 0.3,
        'az': np.concatenate([
            np.linspace(0.1, 0.02, 5),   # Free fall (low)
            np.full(5, 0.02),              # Still falling
            np.linspace(0.02, 2.5, 5) + np.random.randn(5) * 0.5,  # Impact
            np.random.randn(duration - 15) * 0.3 + 1.0  # Recovery
        ]),
        'gx': np.concatenate([
            np.zeros(10),
            np.random.randn(duration - 10) * 100
        ]),
        'gy': np.concatenate([
            np.zeros(10),
            np.random.randn(duration - 10) * 100
        ]),
        'gz': np.concatenate([
            np.zeros(10),
            np.random.randn(duration - 10) * 100
        ]),
    }
    return compute_channels(data)

def compute_channels(data):
    """Compute all 8 channels from raw sensor data."""
    accel_mag = np.sqrt(data['ax']**2 + data['ay']**2 + data['az']**2)
    gyro_mag = np.sqrt(data['gx']**2 + data['gy']**2 + data['gz']**2)
    
    channels = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'accel_mag', 'gyro_mag']
    features = []
    
    for ch in channels:
        if ch == 'accel_mag':
            arr = accel_mag
        elif ch == 'gyro_mag':
            arr = gyro_mag
        else:
            arr = data[ch]
        
        # 8 statistics per channel
        features.extend([
            np.mean(arr),
            np.std(arr),
            np.sum(arr**2),
            np.max(arr),
            np.min(arr),
            np.argmax(np.abs(arr)) / len(arr),
            np.sqrt(np.mean(arr**2)),
            np.max(arr) - np.min(arr)
        ])
    
    return features

# ── Simple Neural Network (NumPy) ──────────────────────────────
class SimpleNN:
    """Lightweight neural network for ESP8266."""
    
    def __init__(self, input_size=64, hidden1=32, hidden2=16):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden1) * np.sqrt(2.0/input_size)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0/hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, 1) * np.sqrt(2.0/hidden2)
        self.b3 = np.zeros(1)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.relu(self.z2)
        self.z3 = self.a2 @ self.W3 + self.b3
        return self.sigmoid(self.z3)
    
    def fit(self, X, y, epochs=100, lr=0.01):
        """Train the network."""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Backpropagation
            delta3 = (output - y.reshape(-1, 1)) * output * (1 - output)
            dW3 = self.a2.T @ delta3
            db3 = np.sum(delta3, axis=0)
            
            delta2 = (delta3 @ self.W3.T) * (self.a2 > 0)
            dW2 = self.a1.T @ delta2
            db2 = np.sum(delta2, axis=0)
            
            delta1 = (delta2 @ self.W2.T) * (self.a1 > 0)
            dW1 = X.T @ delta1
            db1 = np.sum(delta1, axis=0)
            
            # Update weights
            self.W3 -= lr * dW3
            self.b3 -= lr * db3
            self.W2 -= lr * dW2
            self.b2 -= lr * db2
            self.W1 -= lr * dW1
            self.b1 -= lr * db1
            
            if (epoch + 1) % 20 == 0:
                pred = (output > 0.5).astype(int)
                acc = np.mean(pred == y.reshape(-1, 1))
                print(f"Epoch {epoch+1}/{epochs}, Accuracy: {acc:.2%}")

# ── Scaler ─────────────────────────────────────────────────────
class StandardScaler:
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.scale = np.std(X, axis=0)
        self.scale[self.scale == 0] = 1  # Avoid division by zero
        return self
    
    def transform(self, X):
        return (X - self.mean) / self.scale
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)

# ── Generate model.h ───────────────────────────────────────────
def generate_model_header(model, scaler, output_path='model.h'):
    """Generate Arduino-compatible model.h file."""
    
    # Flatten weights
    W1_flat = model.W1.flatten()
    b1_flat = model.b1
    W2_flat = model.W2.flatten()
    b2_flat = model.b2
    W3_flat = model.W3.flatten()
    b3_flat = model.b3
    
    # Architecture: Input(64)→Dense(32)→Dense(16)→Dense(1)
    hidden1 = 32
    hidden2 = 16
    
    with open(output_path, 'w') as f:
        f.write('// ===================================================\n')
        f.write('//   AUTO-GENERATED FILE — DO NOT EDIT MANUALLY\n')
        f.write('//   Generated by fall_detection_model.py\n')
        f.write('// ===================================================\n\n')
        f.write('#ifndef FALL_MODEL_H\n')
        f.write('#define FALL_MODEL_H\n\n')
        f.write('#include <stdint.h>\n\n')
        
        # Model size (weights only)
        total_params = (64 * hidden1 + hidden1 + 
                       hidden1 * hidden2 + hidden2 + 
                       hidden2 * 1 + 1)
        f.write(f'// ── Model size ──────────────────────────────────────\n')
        f.write(f'#define MODEL_SIZE {total_params * 4}\n\n')
        
        # Hidden layer sizes
        f.write(f'// ── Architecture ───────────────────────────────────\n')
        f.write(f'#define HIDDEN1_SIZE {hidden1}\n')
        f.write(f'#define HIDDEN2_SIZE {hidden2}\n\n')
        
        # Weights W1 (64 x 32)
        f.write(f'// ── Layer 1 weights (64 x 32) ─────────────────────\n')
        f.write('const float layer1_weights[] = {\n  ')
        for i, v in enumerate(W1_flat):
            f.write(f'{v:.6f}f, ')
            if (i + 1) % 8 == 0:
                f.write('\n  ')
        f.write('\n};\n\n')
        
        # Bias b1 (32)
        f.write(f'// ── Layer 1 bias (32) ─────────────────────────────\n')
        f.write('const float layer1_bias[] = {\n  ')
        for i, v in enumerate(b1_flat):
            f.write(f'{v:.6f}f, ')
            if (i + 1) % 8 == 0:
                f.write('\n  ')
        f.write('\n};\n\n')
        
        # Weights W2 (32 x 16)
        f.write(f'// ── Layer 2 weights (32 x 16) ─────────────────────\n')
        f.write('const float layer2_weights[] = {\n  ')
        for i, v in enumerate(W2_flat):
            f.write(f'{v:.6f}f, ')
            if (i + 1) % 8 == 0:
                f.write('\n  ')
        f.write('\n};\n\n')
        
        # Bias b2 (16)
        f.write(f'// ── Layer 2 bias (16) ─────────────────────────────\n')
        f.write('const float layer2_bias[] = {\n  ')
        for i, v in enumerate(b2_flat):
            f.write(f'{v:.6f}f, ')
            if (i + 1) % 8 == 0:
                f.write('\n  ')
        f.write('\n};\n\n')
        
        # Weights W3 (16 x 1)
        f.write(f'// ── Output layer weights (16 x 1) ─────────────────\n')
        f.write('const float output_weights[] = {\n  ')
        for i, v in enumerate(W3_flat):
            f.write(f'{v:.6f}f, ')
        f.write('\n};\n\n')
        
        # Bias b3 (1)
        f.write(f'// ── Output layer bias ─────────────────────────────\n')
        f.write(f'const float output_bias = {b3_flat[0]:.6f}f;\n\n')
        
        # Scaler parameters
        f.write(f'// ── Number of features ─────────────────────────────\n')
        f.write(f'#define NUM_FEATURES {NUM_FEATURES}\n\n')
        
        f.write(f'// ── Scaler mean values ─────────────────────────────\n')
        f.write('const float scaler_mean[NUM_FEATURES] = {\n  ')
        for i, v in enumerate(scaler.mean):
            f.write(f'{v:.6f}f, ')
            if (i + 1) % 8 == 0:
                f.write('\n  ')
        f.write('\n};\n\n')
        
        f.write(f'// ── Scaler scale values ─────────────────────────────\n')
        f.write('const float scaler_scale[NUM_FEATURES] = {\n  ')
        for i, v in enumerate(scaler.scale):
            f.write(f'{v:.6f}f, ')
            if (i + 1) % 8 == 0:
                f.write('\n  ')
        f.write('\n};\n\n')
        
        f.write('#endif // FALL_MODEL_H\n')
    
    print(f"\n✓ Model saved to {output_path}")
    return output_path

# ── Main ───────────────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Fall Detection Model Trainer")
    print("=" * 50)
    
    # Generate training data
    print("\n[1] Generating synthetic training data...")
    X, y = generate_synthetic_data(n_samples=2000)
    print(f"    Generated {len(X)} samples ({sum(y)} falls, {len(y)-sum(y)} normal)")
    
    # Normalize features
    print("\n[2] Normalizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train model
    print("\n[3] Training neural network...")
    model = SimpleNN(input_size=64, hidden1=32, hidden2=16)
    model.fit(X_scaled, y, epochs=200, lr=0.1)
    
    # Evaluate
    predictions = (model.forward(X_scaled) > 0.5).astype(int)
    accuracy = np.mean(predictions == y.reshape(-1, 1))
    print(f"\n[4] Final accuracy: {accuracy:.2%}")
    
    # Generate model.h
    print("\n[5] Generating model.h...")
    generate_model_header(model, scaler)
    
    print("\n" + "=" * 50)
    print("  Training complete!")
    print("=" * 50)

if __name__ == "__main__":
    main()