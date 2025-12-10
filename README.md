# Crop Analysis System

## Complete Flow

### Dataset Exploration
We explored multiple datasets including **Plant Village** and **IP102**, but selected the **CCMT Dataset (Cashew, Corn, Maize, Tomato)** sourced from **Kaggle**. This dataset was unique because it provided comprehensive data for both:
-   **Pest Detection**: Included annotated bounding boxes for object localization.
-   **Disease Detection**: High-quality leaf images categorized by disease type.

### Model Experiments
We conducted extensive experiments to find the optimal architecture for each task.

#### 🐛 Pest Detection
For pests, we required object detection (bounding boxes).
-   **Model**: **YOLOv11 (Small)**
-   **Training**: Iterated on the CCMT dataset to accurately localize pests.

#### 🍃 Disease Classification
We experimented with three different architectures for disease diagnosis:

1.  **U-Net (Segmentation)**
    -   Accuracy: ~90%
    -   Result: Dropped (Segmentation was computationally expensive and unnecessary for classification).
2.  **DenseNet (Deep CNN)**
    -   Accuracy: 95-97%
    -   Result: Excellent performance, but slightly heavier.
3.  **EfficientNet-B0 (Efficient CNN)**
    -   Accuracy: **98%**
    -   Result: **Selected as Final Model**. It offered the best trade-off between speed and accuracy.

### Final Model Choices

#### 1. Pest Detection: YOLOv11
-   **Model**: YOLOv11 (Small)
-   **Function**: Detects and localizes pest species on leaves using bounding boxes.

#### 2. Disease Classification: EfficientNet
-   **Model**: EfficientNet-B0
-   **Function**: Classifies crop diseases at the image level with 98% accuracy.

### Deployment Architecture
The complete application is deployed with a modern stack:
-   **Frontend**: Built with **React** for a responsive user interface.
-   **Backend**: Powered by **FastAPI** for high-performance API endpoints and model inference.

---

# Detailed Project Report

## Abstract
Agriculture remains a critical pillar of global food production, where pests and plant diseases represent two of the most damaging biological threats affecting yield output and economic sustainability. Traditional crop diagnosis heavily relies on manual inspection, resulting in delayed treatment and misdiagnosis. This report presents an Artificial Intelligence (AI)-based dual diagnostic system integrating object detection and image classification to simultaneously identify pests through bounding box localization and diagnose leaf diseases in real time. The system employs **YOLOv11** for pest object detection and **EfficientNet-B0** for leaf disease classification using the CCMT dataset. The proposed framework achieves approximately **0.85 mAP@0.5** for pest localization and nearly **98% accuracy** for disease classification, demonstrating strong practical potential for early detection and farmer decision support.

## 1. Introduction
Agriculture is constantly exposed to biological threats, primarily caused by pests and plant pathogens. These threats directly affect crop health and productivity, leading to reduced yield quantity and significant financial losses for farmers. Early-stage pest activity often goes unnoticed due to their small size, natural camouflage, and rapid movement. Similarly, visual symptoms of leaf diseases during initial stages appear highly similar, making manual diagnosis difficult and error-prone.

Deep learning technologies have recently demonstrated impressive capabilities in agricultural image analysis. However, most existing systems focus exclusively on either classification or detection. This project addresses this limitation by developing a unified framework that performs both tasks simultaneously. By offering real-time inference and visual explanations, the system provides practical value for farmers and agricultural advisors.

## 2. Problem Statement
Traditional crop diagnosis methods experience multiple limitations:
*   Rely on costly expert knowledge
*   Manual inspection delays treatment
*   Focus on only one type of threat
*   No pest localization output
*   High probability of human error

Early detection is essential for minimizing crop damage; therefore, a dual diagnostic system becomes necessary for modern precision agriculture.

## 3. Objectives
The primary objectives of the proposed system are:
*   Detect pests and localize them using bounding boxes
*   Classify leaf diseases accurately
*   Provide real-time inference for field use
*   Maintain lightweight design suitable for edge deployment

## 4. Methodology

### 4.1 YOLOv11 Pest Detection
YOLOv11 provides real-time object detection through single-stage detection architecture. The **Small** variant was used, offering compact size and fast inference. The model produces bounding boxes along with confidence scores marking exact pest locations on crop images.

### 4.2 EfficientNet-B0 Classification
EfficientNet scales depth, width, and resolution in a compound manner. Three architectures were tested:
*   **U-Net**: Segmentation, unnecessary complexity.
*   **DenseNet201**: Strong accuracy, more computational load.
*   **EfficientNet-B0**: Best accuracy and efficiency.

EfficientNet-B0 achieved **98% accuracy** and outperformed alternative approaches while remaining operational for real-time classification.

## 5. Results

### 5.1 YOLOv11 Pest Detection
*   **mAP@0.5**: 0.85
*   **mAP@0.5–0.95**: 0.52
*   **Precision**: 0.84
*   **Recall**: 0.80

These results reflect strong localization ability even for small and camouflaged pests.

### 5.2 EfficientNet-B0 Disease Classification
*   **Accuracy**: ≈ 98%
*   **Latency**: Low inference latency
*   **Hardware**: Minimal GPU requirements

## 6. Conclusion
The proposed AI system effectively detects biological vectors and plant diseases simultaneously. Combining **YOLOv11** and **EfficientNet-B0** results in strong detection performance and high classification accuracy. This dual-engine framework supports early intervention, provides visual evidence to farmers, and helps prevent irreversible crop failure.
