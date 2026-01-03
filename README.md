# Speech Emotion Recognition Using Deep Learning

A deep learning project that classifies emotions from speech audio using Convolutional Neural Networks (CNN). The model achieves **93.54%** accuracy on the test dataset by combining multiple datasets and applying advanced feature extraction and data augmentation techniques.

## 📊 Project Overview
This project implements an end-to-end speech emotion recognition system that can classify audio samples into 7 different emotions:
- 😠 Angry
- 🤢 Disgust
- 😨 Fear
- 😊 Happy
- 😐 Neutral
- 😢 Sad
- 😲 Surprise

## 🎯 Model Performance
- **Test Accuracy**: 93.54%
- **Architecture**: Convolutional Neural Network (CNN)
- **Training Time**: ~40 minutes (with GPU acceleration)
- **Dataset Size**: 12,162 audio files (48,648 samples after augmentation)

### Classification Report

| Emotion  | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Angry    | 0.95      | 0.93   | 0.94     | 1577    |
| Disgust  | 0.94      | 0.93   | 0.94     | 1538    |
| Fear     | 0.93      | 0.92   | 0.92     | 1555    |
| Happy    | 0.91      | 0.94   | 0.92     | 1468    |
| Neutral  | 0.96      | 0.94   | 0.95     | 1515    |
| Sad      | 0.92      | 0.95   | 0.93     | 1523    |
| Surprise | 0.94      | 0.92   | 0.93     | 554     |
| **Overall** | **0.94** | **0.94** | **0.94** | **9730** |

## 📁 Datasets Used

The model is trained on four popular emotional speech datasets:

1. **RAVDESS** (Ryerson Audio-Visual Database of Emotional Speech and Song)
   - 1,440 audio files
   - 24 professional actors (12 male, 12 female)
   - [Dataset Link](https://www.kaggle.com/datasets/uwrfkaggle/ravdess-emotional-speech-audio)

2. **CREMA-D** (Crowd-Sourced Emotional Multimodal Actors Dataset)
   - 7,442 audio files
   - 91 actors with diverse ethnicities
   - [Dataset Link](https://www.kaggle.com/datasets/ejlok1/cremad)

3. **TESS** (Toronto Emotional Speech Set)
   - 2,800 audio files
   - 2 female actors (aged 26 and 64)
   - [Dataset Link](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)

4. **SAVEE** (Surrey Audio-Visual Expressed Emotion)
   - 480 audio files
   - 4 male native English speakers
   - [Dataset Link](https://www.kaggle.com/datasets/ejlok1/surrey-audiovisual-expressed-emotion-savee)

## 🔧 Features & Techniques

### Feature Extraction
- **MFCC** (Mel-frequency cepstral coefficients)
- **Zero Crossing Rate** (ZCR)
- **Root Mean Square Energy** (RMSE)

### Data Augmentation
To increase dataset size and model robustness, we applied 4 augmentation techniques:
1. **Noise Injection**: Adding random noise to audio
2. **Time Stretching**: Changing audio speed without affecting pitch
3. **Pitch Shifting**: Altering the pitch of audio
4. **Combined**: Pitch shifting + Noise injection

This increased our dataset from 12,162 to 48,648 samples (4x augmentation).

### Model Architecture

```
CNN Model Architecture:
├── Conv1D (512 filters, kernel=5) + BatchNorm + MaxPool
├── Conv1D (512 filters, kernel=5) + BatchNorm + MaxPool + Dropout(0.2)
├── Conv1D (256 filters, kernel=5) + BatchNorm + MaxPool
├── Conv1D (256 filters, kernel=3) + BatchNorm + MaxPool + Dropout(0.2)
├── Conv1D (128 filters, kernel=3) + BatchNorm + MaxPool + Dropout(0.2)
├── Flatten
├── Dense (512 units, ReLU) + BatchNorm
└── Dense (7 units, Softmax)

Total Parameters: 7,193,223
Trainable Parameters: 7,188,871
```

### Training Configuration
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 20 (with early stopping)
- **Callbacks**: 
  - Early Stopping (patience=5)
  - Learning Rate Reduction (patience=3, factor=0.5)
  - Model Checkpoint (saves best weights)

## 🚀 Getting Started

### Prerequisites

```bash
python >= 3.8
tensorflow >= 2.x
librosa
pandas
numpy
scikit-learn
matplotlib
seaborn
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Abdulsalam223/speech-emotion-recognition.git
cd speech-emotion-recognition
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

3. Download the datasets from Kaggle (links provided above)

### Usage

#### Training the Model

Run the complete training pipeline in Kaggle or Jupyter Notebook:

```python
# The notebook contains all steps from data preprocessing to model training
# Simply run all cells sequentially
```

#### Making Predictions

```python
from tensorflow.keras.models import model_from_json
import pickle
import librosa
import numpy as np

# Load the trained model
json_file = open('CNN_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("best_model1_weights.h5")

# Load preprocessing objects
with open('scaler2.pickle', 'rb') as f:
    scaler = pickle.load(f)
with open('encoder2.pickle', 'rb') as f:
    encoder = pickle.load(f)

# Define prediction function
def predict_emotion(audio_path):
    # Load audio
    data, sr = librosa.load(audio_path, duration=2.5, offset=0.6)
    
    # Extract features
    result = extract_features(data, sr)
    result = np.reshape(result, newshape=(1, 2376))
    
    # Scale features
    result = scaler.transform(result)
    result = np.expand_dims(result, axis=2)
    
    # Predict
    predictions = loaded_model.predict(result)
    emotion = encoder.inverse_transform(predictions)
    
    return emotion[0][0]

# Use the function
emotion = predict_emotion("path/to/audio.wav")
print(f"Predicted Emotion: {emotion}")
```

## 📊 Results Visualization

### Training History
The model shows consistent improvement across epochs with minimal overfitting:

- **Training Accuracy**: 92.94%
- **Validation Accuracy**: 93.54%
- **Training Loss**: 0.2040
- **Validation Loss**: 0.1992

### Confusion Matrix
The confusion matrix shows strong diagonal values, indicating accurate predictions across all emotion classes with minimal misclassification.

## 📂 Project Structure

```
speech-emotion-recognition/
├── notebook.ipynb              # Main Jupyter notebook with complete pipeline
├── CNN_model.json              # Model architecture (JSON format)
├── best_model1_weights.h5      # Trained model weights
├── scaler2.pickle              # StandardScaler for feature normalization
├── encoder2.pickle             # OneHotEncoder for label encoding
├── emotion.csv                 # Extracted features dataset
├── data_path.csv              # Combined dataset paths and labels
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

## 🔍 Key Insights

1. **Balanced Performance**: The model performs consistently well across all emotion classes with F1-scores ranging from 0.92 to 0.95.

2. **Neutral Emotion**: Achieved the highest precision (0.96), making it the most reliably detected emotion.

3. **Data Augmentation Impact**: 4x data augmentation significantly improved model generalization and robustness.

4. **CNN Effectiveness**: The CNN architecture effectively captures temporal patterns in audio features for emotion classification.

5. **Multi-Dataset Training**: Combining four diverse datasets improved model generalization across different speakers, ages, and recording conditions.

## 🛠️ Technologies Used

- **TensorFlow/Keras**: Deep learning framework
- **Librosa**: Audio processing and feature extraction
- **Scikit-learn**: Data preprocessing and evaluation metrics
- **Pandas & NumPy**: Data manipulation
- **Matplotlib & Seaborn**: Data visualization
- **Joblib**: Parallel processing for faster feature extraction

## 💡 Future Improvements

- [ ] Implement real-time emotion detection from microphone input
- [ ] Add LSTM layers for better temporal sequence modeling
- [ ] Experiment with transfer learning using pre-trained audio models
- [ ] Deploy as a web application using Flask/Streamlit
- [ ] Add multilingual emotion recognition support
- [ ] Implement ensemble methods combining CNN and LSTM
- [ ] Create a mobile app for emotion detection

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- RAVDESS, CREMA-D, TESS, and SAVEE dataset creators
- Kaggle community for dataset hosting and computational resources
- TensorFlow and Librosa development teams

## 👨‍💻 Author

**Abdul Salam**
- GitHub: [@Abdulsalam223](https://github.com/Abdulsalam223)
- LinkedIn: [abdulsalam001](https://linkedin.com/in/abdulsalam001)

## 📧 Contact

For questions or collaboration opportunities, feel free to reach out:
- Email: khanabd22385@gmail.com
- Issues: [GitHub Issues](https://github.com/Abdulsalam223/speech-emotion-recognition/issues)

## ⭐ Show Your Support

If you found this project helpful, please give it a ⭐ on GitHub!

---

**Note**: This project was developed as part of a machine learning/deep learning study in audio emotion recognition. The model is for educational and research purposes.
