

---
# UCI Digit Classification

## Overview
This repository contains a Java implementation of machine learning classifiers for handwritten digit recognition using the UCI Optical Recognition of Handwritten Digits dataset. The project was developed as coursework to explore different classification approaches from scratch.

It includes the `UCI_CW` folder, which holds:

- **AI Report**: Detailed explanation of the project methodology and results (`Koller Melanie AI Report.pdf`)  
- **Java Project** `Digit_Classification` Eclipse project:
  - `src/` – source code
  - `bin/` – compiled binaries
  - `datasets/` – CSV datasets (dataset1 and dataset2) for two-fold 

The system implements a k-Nearest Neighbors baseline algorithm and was extended with more advanced classifiers including Support Vector Machine and an experimental Multi-Layer Perceptron. 


---
### Project Structure

      Digit_classification/
      ├── datasets/
      │   ├── dataset1.csv
      │   └── dataset2.csv
      ├── src/
      │   ├── combinedMain/
      │   │   └── CombinedDigitClassifier.java
      │   ├── common/
      │   │   ├── data/
      │   │   │   ├── DataPoint.java
      │   │   │   └── DatasetLoader.java
      │   │   ├── evaluation/
      │   │   │   ├── ConfusionMatrix.java
      │   │   │   └── ResultsAnalyzer.java
      │   │   └── utils/
      │   │       └── Statistics.java
      │   ├── knnModel/
      │   │   ├── DistanceMetric.java
      │   │   ├── KnnClassifier.java
      │   │   └── MainKnn.java
      │   ├── mlpModel/
      │   │   ├── Activation.java
      │   │   ├── Layer.java
      │   │   ├── MainMlp.java
      │   │   └── MlpClassifier.java
      │   └── svmModel/
      │       ├── Kernel.java
      │       ├── MainSvm.java
      │       └── SvmClassifier.java
      └── Koller Melanie AI Report.pdf


---

## Features

1. k-Nearest Neighbors (k-NN) with Euclidean/Manhattan distance metrics
2. Support Vector Machine (SVM) with multiple kernel functions
3. Multi-Layer Perceptron (MLP) neural network (experimental)
4. Two-Fold Cross-Validation for robust evaluation
5. Comprehensive Results Analysis including confusion matrices
      

---

## How to Run
1. Clone or download this repository
2. Open Eclipse IDE
3. Import the project: File → Import → Existing Projects into Workspace
4. Select the Digit_classification folder
5. Ensure the datasets/ folder is correctly linked in the project build path
6. Run any of the following main classes:
   
    1. CombinedDigitClassifier.java - Runs all models
    2. MainKnn.java - Comprehensive k-NN evaluation
    3. MainSvm.java - Comprehensive SVM evaluation
    4. MainMlp.java - MLP model (experimental)

---
### Dataset Information

**Source**: UCI Optical Recognition of Handwritten Digits

**Format**: 8×8 pixel images of digits (0-9)

**Preprocessing**: NIST methodology applied

**Splits**: Two datasets for two-fold cross-validation

**Features**: 64 normalized integer values (0-16 range)

---
## Technologies

**Language**: Java (Standard Edition)

**Development Environment**: Eclipse IDE

**Libraries**: Standard Java only (no external dependencies)

**Version Control**: Git

---

## License

This project is for **educational purposes** as part of the UCI coursework and is not intended for commercial use.

---
## Author

**Melanie Koller**

University Coursework Project

Artificial Intelligence Module
