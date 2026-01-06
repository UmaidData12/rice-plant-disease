# 🌾 Rice Plant Disease Detection

A deep learning application for detecting rice plant diseases using CNN with 89.93% accuracy. Upload a rice leaf image to get instant predictions and disease management recommendations.

## 🎯 Features
- **6 Disease Detection**: Bacterial Leaf Blight, Brown Spot, Leaf Blast, Leaf Scald, Sheath Blight, and Healthy Leaf Classification
- **High Accuracy**: 89.93% validation accuracy
- **Real-time Predictions**: Instant results with confidence scores
- **Comprehensive Disease Info**: Detailed cause, symptoms, and management strategies for each disease
- **User-Friendly Interface**: Modern, responsive Streamlit web application
- **Fast Processing**: Optimized for quick predictions on standard hardware

## 📁 Project Structure
```
Rice Disease/
├── leaf_app.py                 # Main Streamlit application
├── rice_plant_disease.ipynb    # Jupyter notebook with model training & analysis
├── requirements.txt            # Python dependencies
├── README.md                  # This file
└── models/
    └── cnn_best_model.h5      # Pre-trained CNN model
```

## 📋 System Requirements
- Python 3.8 or higher
- At least 2GB RAM recommended
- 500MB free disk space for model and dependencies

## 🚀 Installation & Setup

### Option 1: Local Setup (Recommended for Development)

#### Step 1: Navigate to project directory
```bash
cd "path/to/Rice Disease"
```

#### Step 2: Create a virtual environment (optional but recommended)
```bash
python3 -m venv venv
source venv/bin/activate
```

#### Step 3: Install Python dependencies
```bash
pip install -r requirements.txt
```

#### Step 4: Download the Pre-trained Model
The model file is hosted separately due to size constraints:

1. **Download** `cnn_best_model.h5` from this link:
   📥 [Download Model from Google Drive](https://drive.google.com/file/d/1KMii1LyyVjjt-gO9eKx0L09CGNGKa2lN/view?usp=sharing)

2. **Extract** the downloaded file

3. **Place** it in the `models/` folder:
   ```
   Rice Disease/
   └── models/
       └── cnn_best_model.h5  ← Place the model file here
   ```

4. **Verify** the folder structure matches before running the app

#### Step 5: Run the application
```bash
streamlit run leaf_app.py
```

The app will open automatically at `http://localhost:8501` in your browser.

---

### Option 2: Docker Setup (For Production/Deployment)

#### Step 1: Build the Docker image
```bash
docker build -t rice-disease-detector .
```

#### Step 2: Run the container
```bash
docker run -p 8501:8501 rice-disease-detector
```

Access the app at `http://localhost:8501`

---

## 💻 How to Use

1. **Upload an Image**
   - Click "Drag and drop file here" or "Browse files"
   - Supported formats: JPG, JPEG, PNG, BMP
   - Maximum file size: 200MB

2. **Get Prediction**
   - Click the "PREDICT DISEASE" button
   - Wait for the analysis (usually < 2 seconds)

3. **View Results**
   - Disease name and confidence percentage
   - Disease cause explanation
   - Symptoms to watch for
   - Management and treatment recommendations

4. **Try Another Image**
   - Click "DETECT ANOTHER IMAGE" to upload and analyze a new image

## 🤖 Model Information

| Property | Value |
|----------|-------|
| **Architecture** | CNN with Data Augmentation & Batch Normalization |
| **Input Size** | 128×128 RGB images |
| **Validation Accuracy** | 89.93% |
| **Total Parameters** | 2.78M |
| **Training Framework** | TensorFlow/Keras |
| **Model File** | `cnn_best_model.h5` |

## � Dataset

The dataset used for training this model contains augmented rice leaf images across 6 disease categories:

📥 **Download Dataset**: [Rice Leaf Dataset (OneDrive)](https://1drv.ms/u/c/80072641619f0550/EdIwaNG2_GNDrDux3l0AyEUBt0lpZJDJIP1ZKa9cOT91Lg)

Extract the dataset and update the `dataset_dir` path in `rice_plant_disease.ipynb` before retraining.

## �🔬 Training & Analysis

For detailed model training, data preprocessing, and analysis, see `rice_plant_disease.ipynb`:
```bash
jupyter notebook rice_plant_disease.ipynb
```

## 🛠️ Technologies Used

| Technology | Purpose |
|-----------|---------|
| **TensorFlow/Keras** | Deep learning framework & CNN model |
| **Streamlit** | Web application framework |
| **NumPy** | Numerical computations |
| **Pillow (PIL)** | Image processing |
| **OpenCV** | Computer vision utilities |
| **Pandas** | Data manipulation (training) |
| **Matplotlib** | Visualization (training) |

## 📦 Dependencies

**Python Packages** (see `requirements.txt`):
- streamlit >= 1.28.0
- numpy >= 1.24.0
- pandas >= 2.0.0
- matplotlib >= 3.7.0
- opencv-python-headless >= 4.8.0
- tensorflow >= 2.13.0
- Pillow >= 10.0.0

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found error | Ensure `models/cnn_best_model.h5` exists in the project folder |
| ImportError (TensorFlow) | Run `pip install --upgrade tensorflow` |
| Port 8501 already in use | Run `streamlit run leaf_app.py --server.port 8502` |
| Slow predictions | Increase available RAM or run on a machine with GPU support |

## 📝 License

MIT License - Feel free to use, modify, and distribute this project.

## 👨‍💼 Author

Umed Ali (FYP Project - Rice Plant Disease Detection)


