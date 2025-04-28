# 📚 Image Analysis Computational Project

**University**: University of Piraeus  
**Department**: Informatics  
**Academic Year**: 2023-2024  
**Semester**: 7th

**Course**:  
"Image Analysis"  
**Project**: Computational Assignment  
**Submission Date**: 12.02.2024

---

## 🛠 Project Setup

The programs were developed in **Visual Studio Code** using **Jupyter Notebook**.  
The code includes **concise and meaningful comments** for better understanding.

The materials provided are:
- 📄 A PDF file (project report)
- 🐍 Python code (.py)
- 📓 Jupyter Notebook (.ipynb)
- 🌐 An HTML file converted from the Notebook, showing a full execution of the algorithm with results and comments.

---

## 📖 Introduction

This project implements a **graph theory-based algorithm** for **content-based image retrieval (CBIR)**, following the principles described in the paper *"Multimedia Retrieval through Unsupervised Hypergraph-based Manifold Ranking."*

---

## 🔎 Overview of the Algorithm

The proposed method, called **"Log-Based Hypergraph of Ranking Reference"**, is executed in five main steps:

- **Feature Extraction**:  
  A pretrained model from `torchvision` (Vision Transformer) was used to extract features from images by removing the classifier layer and using the last hidden layer's output.

- **Key Concepts**:  
  We work with a set of multimedia objects/images. For each query image, the algorithm retrieves the top-**k** most similar images based on a similarity function (in our case, the inverse of the Euclidean distance).

- **Steps**:
  1. **Ranking Normalization**:  
     We normalize the ranking lists to ensure symmetry between neighbor relationships.
  
  2. **Hypergraph Construction**:  
     A hypergraph is built where each hyperedge connects an image with its **k-nearest neighbors**, using a probabilistic participation measure.
  
  3. **Hyperedge Similarity Computation**:  
     Using incidence matrices and a Hadamard product, we calculate the similarity between hyperedges.

  4. **Cartesian Product Calculation**:  
     We perform Cartesian products of hyperedges to compute pairwise relationships between images.

  5. **Similarity Evaluation**:  
     A final affinity matrix **W** is constructed, combining all the previous computations, leading to a refined image ranking.

---

## 🧩 Implementation Details

- Programming Language: **Python** 🐍
- Environment: **Jupyter Notebook** 📓
- Key Libraries Used:
  - `torch`, `torchvision`
  - `numpy`
  - `matplotlib`
  - `random`
  - `timm`
  - `gdown` (for downloading resources)

- **Dataset**:  
  Subset of **Caltech101** 📷, manually downloaded and integrated into the project folder.

- **Feature Extraction**:  
  We used a **Vision Transformer** (`vit_base_patch16_224`) pretrained on **ImageNet**, replacing its classifier with an identity layer to get the feature vectors.

- **Additional Helper Script**:  
  A helper Python file was used for visualization, sourced from an external GitHub repository.

---

## ✨ Key Notes

- We don't specify fixed query images.  
  Instead, **all images** can act as a query, allowing flexibility in retrieval evaluation.
  
- A **N x N** similarity matrix is computed, where **N** is the number of images.

- Each row represents the similarity scores of a given image to all others.

---

## 📷 Example Executions

The project includes **7 examples** showing query images along with their retrieved results and corresponding **precision** and **recall** scores.

Each example lists:
- The **query image** 🖼️
- **Retrieved images** ranked by similarity 🔍
- Calculated **Precision** 📈
- Calculated **Recall** 📊

![Screenshot 2025-04-28 at 3 42 44 PM](https://github.com/user-attachments/assets/87391129-2edc-4c34-a0ec-e3298943cd4f)

---

## 🚀 Conclusion

This project successfully implements a hypergraph-based approach for content-based image retrieval, showcasing the practical use of advanced deep learning models for feature extraction and graph theory for retrieval.
