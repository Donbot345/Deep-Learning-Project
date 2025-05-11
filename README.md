# Deep-Learning-Project
YOLOv5 Object Detection Project: 
Implementing YOLOv5 for the automated detection of plant diseases using the PlantDoc dataset with 30 disease classes. 

## Project Overview 
The project implements YOLOv5 to detect and classify plant disease across different crops and plants. The system identifies 30 disease classes and provides a bounding box localization and classification. 

## Key Features 
- Automated detection of 30 plant disease classes
- Fast inference time is about 6.4ms per image
- Provides comprehensive class distribution analysis
- Performance evaluation across disease categories
- The entire deployment model (14.1MB)

## Dataset Summary
The project uses the PlantDoc dataset from Kaggle:
- Source: https://www.kaggle.com/datasets/andresmgs/plantdec
- Classes: 30 Disease classes across various plant species
- Total Images
    - Training: 1979
    - Validation: 349
    - Test: 239 
- Bounding Box annotations in YOLO format
- RGB images of plant leaves with disease symptoms

## Repository structure 
├── README.md

├── finished.py

├── report 

├── presentation 

├── models 

├── visualizations 

└── requirements 

## Setup and Installation 
1. Clone the repository:
!git clone https://github.com/ultralytics/yolov5.git
%cd yolov5
!pip install -qr requirements.txt

2. Mount Drive and Extract
from google.colab import drive
zip_path = '/content/drive/MyDrive/plantdec.zip'
drive.mount('/content/drive')
!unzip -q -o "{zip_path}" -d /content/plantdec

3. Train Model
  !python train.py \
  --img 640 \
  --batch 16 \
  --epochs 30 \
  --data /content/plantdec/data.yaml \
  --weights yolov5s.pt \
  --name plantdec_yolov5s3 \
  --cache --workers 4 --patience 10 --optimizer Adam \
  --project /content/plantdec_results

4. Run Inference
!python detect.py \
  --weights /content/plantdec_results/plantdec_yolov5s3/weights/best.pt \
  --source /content/plantdec/test/images \
  --img 640 \
  --conf 0.25 \
  --name test_predictions \
  --project inference_output
   
## Class Distribution: 
The dataset displays significant class imbalance:
- Most represented: Blueberry Leaf (838 instances)
- Least represented: Tomato two-spotted spider mites leaf (2 instances)

## Implementation Details 
- Model: YOLOv5s (small variant)
- Image Size: 640x640
- Batch Size: 16
- Epochs: 30
- Optimizer: Adam
- Loss Functions:
    - Box Loss, Objectness Loss, Classification Loss
- Evaluation Metrics:
    - Precision, Recall, mAP@0.5, mAP@0.50.95
- Architecture: CSP backbone + PANet neck + Detection heads

## Training Curves 
Each curve shown tracks the respective metric over the 30 training epochs:
- Box Loss: Decreasing trend
- Objectness Loss: Steady decreasing trend
- Classification Loss: Stabilized
- Precision: Steadily increasing
- Recall: Fluctuation
- mAP@0.5: Steadily increasing
  
## Training Metrics (Final Epoch) 
- Precision: 0.51
- Recall: 0.25
- mAP@0.5: 0.22
- mAP@0.50.95: 0.14

## Inference Results 
Inference was run on six randomly selected test images. The model was able to detect diseases from varying images with different lighting and orientations, with bounding boxes and class confidence scores. They were displayed using matplotlib and PIL from the prediction outputs.

## Key Findings 
1. Class imbalance significantly impacted the performance, with underrepresented classes showing lower detection rates
2. YOLOv5s provides a great balance of speed and accuracy for potential field deployment
3. Disease patterns with distinctive visual characteristics achieve higher detection accuracy
4. Multi-scale detection capability that can effectively handle varying lesion sizes.

## Future Work 
- Implement class weights and loss to help address class imbalance
- Test larger YOLOv5 variants (m/l) to improve accuracy
- Apply targeted data augmentation for underrepresented classes
  
## Conclusion 
The implementation of YOLOv5 for plant disease detection demonstrates the viability of deep learning object detection for agricultural applications. The project successfully trained a YOLOv5 model to detect and classify 30 different plant disease classes and achieved an overall precision of 0.51, recall: 0.25, mAP@0.5: 0.22, and mAP@0.50.95: 0.14, with an inference speed of 6.4ms per image. 

The analysis revealed significant variation in performance across different disease classes. There was a clear correlation between class representation in the training dataset and detection accuracy. So high-performing classes like Corn rust leaf (mAP@0.5: 0.723) typically featured distinctive visual characteristics and abundant training samples, while underperforming classes like Tomato Early blight leaf (mAP@0.5: 0.027) suffered from visual similarity to other diseases and limited representation in the dataset.

The implementation emphasizes the essential balance between accuracy and efficiency needed for practical agricultural applications. The compact YOLOv5s model, which is only 14.1 MB, offers real-time inference capabilities that are suitable for field deployment on resource-constrained devices while still delivering reasonable detection performance. This balance makes the system practical for real-world use, where timely detection can significantly influence disease management results.

Class imbalance has emerged as a significant challenge, with the model displaying a bias toward well-represented classes. This finding highlights the importance of balanced datasets in agricultural applications, especially since some diseases are naturally rarer but equally important to detect. Future implementations should tackle this issue by focusing on targeted data collection, employing augmentation strategies, and modifying loss functions.

Overall, the project lays a solid foundation for automated plant disease detection, presenting clear avenues for improvement and practical deployment. By facilitating early and accurate detection of plant diseases, such systems can greatly contribute to sustainable agriculture by minimizing crop losses, optimizing treatment applications, and enhancing integrated pest management strategies. These deep learning-based detection systems can potentially transform disease monitoring and management practices worldwide.

## Citations 
1. YOLOv5 (Object Detection Framework)
Ultralytics. (2020). YOLOv5. GitHub repository.
https://github.com/ultralytics/yolov5

2. PlantDoc Dataset
MGS, A. (2023). PlantDoc - Plant Disease Dataset. Kaggle.
https://www.kaggle.com/datasets/andresmgs/plantdec

3. PyTorch (Deep Learning Framework)
Paszke, A., Gross, S., Massa, F., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.
https://pytorch.org/

4. Google Colab (Notebook Environment)
Google Research. Google Colaboratory.
https://colab.research.google.com/

5. Roboflow (Dataset Formatting & YAML Conversion)
Roboflow. (n.d.). Roboflow: Organize, Label, and Export Datasets for YOLOv5 and Other Models.
https://roboflow.com/

6. Matplotlib & Seaborn (Visualization Libraries)
Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering.
Waskom, M. L. (2021). Seaborn: Statistical data visualization. Journal of Open Source Software.
https://matplotlib.org/
https://seaborn.pydata.org/

7. Pandas (Data Handling)
McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference.
https://pandas.pydata.org/

[YOLOv5 - Ultralytics](https://github.com/ultralytics/yolov5) 
[PlantDoc Dataset - Roboflow](https://universe.roboflow.com/joseph-nelson/plantdoc)
[PlantDoc Dataset - Kaggle]((https://www.kaggle.com/datasets/andresmgs/plantdec))
[Roboflow Annotation Tools](https://roboflow.com/)
[Matplotlib](https://matplotlib.org/) and [Seaborn](https://seaborn.pydata.org/)
[Pandas](https://pandas.pydata.org/)

## License 
- MIT

## Author 
- Dante Quinones 
