# Python Tools for Fall Detection Model Training

## Files

1. **fall_detection_model.py** - Train a model from scratch using synthetic data
2. **convert_to_header.py** - Convert a trained Keras model (.h5) to Arduino header

## Installation

Install required Python packages:

```bash
pip install numpy tensorflow scikit-learn
```

## Usage

### Option 1: Train from scratch (synthetic data)

```bash
cd python_tools
python fall_detection_model.py
```

This generates a `model.h` file with trained weights.

### Option 2: Convert existing Keras model

```bash
python convert_to_header.py --model mymodel.h5 --output ../fall_detection/model.h
```

### Option 3: Train with real sensor data

1. Collect data using `DATASET_MODE = true` in `config.h`
2. Save the CSV output
3. Modify `fall_detection_model.py` to load your CSV data
4. Run training and generate new `model.h`

## Requirements

- Python 3.7+
- numpy
- tensorflow (for Keras model conversion)
- scikit-learn (for scaler)