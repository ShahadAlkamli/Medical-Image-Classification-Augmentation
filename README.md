# Medical Image Classification with Data Augmentation  
### Improving Skin Cancer Detection Using Traditional Machine Learning Models

This project explores the impact of **data augmentation** on **medical image classification**, specifically for **melanoma (malignant) vs. benign skin lesions**.  
The pipeline includes:

- Data augmentation using `ImageDataGenerator`  
- Merging raw and augmented datasets  
- Image preprocessing and resizing  
- Dimensionality reduction using PCA  
- Classification using SVM, Decision Tree, Random Forest, and Naive Bayes  
- Performance evaluation before and after augmentation

---

## 📁 Project Structure

```
Medical-Image-Classification-Augmentation/
│
├── augmented_images/               # Generated augmented images (benign/malignant)
│
├── code/
│     ├── Augmentation.py           # Code for generating augmented images
│     ├── Merge.py                  # Combines raw + augmented datasets
│     └── Evaluation.py             # PCA + ML classifiers + evaluation
│
├── results_and_analysis/
│     ├── Results before augmentation.pdf
│     └── Results after augmentation.pdf
│
└── README.md
```

---

## 🧬 Dataset

Two classes are used:

- **benign**
- **malignant**

Images are augmented to artificially increase dataset size and improve model generalization.

---

## 🧪 1. Data Augmentation

The augmentation script (`Augmentation.py`) applies transformations such as:

- Rotation (40°)  
- Width/height shifts  
- Shearing  
- Zooming  
- Horizontal flips  
- Nearest-pixel fill  

These transformations enrich the dataset and reduce overfitting.

Example (from your code):

```python
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
```

Augmented images are stored in:

```
augmented_images/
    ├── benign/
    └── malignant/
```

---

## 🗂️ 2. Dataset Merging

`Merge.py` merges:

- Original training images  
- Augmented images  

into a single folder: `augmented_train/`

This ensures balanced classes and increased training data.

---

## 🖼️ 3. Preprocessing & PCA

`Evaluation.py` performs:

- Image loading using `skimage`
- Resizing to 100×100 pixels
- Flattening pixel arrays
- PCA reduction to 100 components

PCA dramatically reduces dimensionality while retaining the most important variance.

---

## 🤖 4. Machine Learning Models

Four traditional classifiers are trained:

| Model | Type |
|-------|------|
| SVM | Support Vector Machine (Linear Kernel) |
| Decision Tree | CART |
| Random Forest | Ensemble of Trees |
| Naive Bayes | Gaussian NB |

---

## 📊 5. Results

### **Before Augmentation**
From *Results before augmentation.pdf*:

- **SVM Accuracy:** 0.862  
- **Random Forest Accuracy:** 0.887 (best model)  
- **Naive Bayes:** 0.826  
- **Decision Tree:** 0.841  

Confusion matrices (page 2) show strong performance for Random Forest.

---

### **After Augmentation**
From *Results after augmentation.pdf*:

- **SVM Accuracy:** 0.871  
- **Random Forest Accuracy:** 0.885  
- **Decision Tree:** 0.847  
- **Naive Bayes:** 0.793  

SVM and Decision Tree improved slightly.  
Naive Bayes decreased in accuracy due to noise sensitivity.  
Random Forest remained the strongest performer (0.885).  
See confusion matrices on page 2.

---

## 📈 Accuracy Comparison

### **Before Augmentation (Best: Random Forest 0.887)**  
### **After Augmentation (Best: Random Forest 0.885)**  

Augmentation improved stability but did not surpass the best pre-augmentation RF accuracy.  
However, it improved class balance and model robustness.

---

## ▶️ How to Run the Code

### **1. Generate Augmented Images**
```bash
python Augmentation.py
```

### **2. Merge Raw + Augmented Data**
```bash
python Merge.py
```

### **3. Train Models & Evaluate**
```bash
python Evaluation.py
```

This produces:

- Classification reports  
- Confusion matrices  
- Accuracy comparison plots  

---

## 🧠 Key Findings

- **Random Forest** consistently achieved the highest accuracy (0.885–0.887).  
- **SVM** improved after augmentation (0.862 → 0.871).  
- Augmentation increased dataset diversity and reduced overfitting for some models.  
- Naive Bayes was negatively affected by augmentation noise.

---

## 📜 License

This project is for educational and research purposes.

