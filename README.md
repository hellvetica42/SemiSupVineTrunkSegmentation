# Project Structure
This project contains code developed during the research of the "Generating vine trunk semantic segmentation dataset via
semi-supervised learning and object detection" paper pending peer review.

* SSL_ELN folder contains slightly modified code from the original [Semi-supervised Semantic Segmentation with Error Localization Network](https://github.com/kinux98/SSL_ELN) project. It's used for training the segmentation decoder with a limited annotated dataset and a large unannotated dataset.
* LABELED_DATA folder contains about 300 labeled thumbnails of vine trunks exported from [RoboFlow](https://roboflow.com/)
* vineset folder contains only the structure of the vineset dataset. The original dataset can be found [here](https://zenodo.org/records/5362354) 
* TrainPytorch.ipynb is a notebook containing the training process and evaluation of the Segmentation Models based on the [Segmentation models with pretrained backbones. PyTorch](https://github.com/qubvel/segmentation_models.pytorch) project.
* SEG_MODELS folder is meant to contain trained checkpoints of Segmentation Models trained for vine trunk segmentation. However, the files are too large for git lfs. If you want access to the trained models feel free to contact me.
* EDA.ipynb is a notebook containing Exploratory Data Analysis of the generated dataset and some filtering.
* EvaluatePerformance.ipynb is a notebook that evaluates the processing speed of each model and draws a Processing time vs Accuracy graph.
* DETECT_AND_SEGMENT folder contains the final detection and segmentation process. This includes the trained yolov5 model (best.pt) on the vineset dataset.