
# DETECTING PNEUMONIA CASE IN CHEST X-RAY IMAGES USING DEEP LEARNING TO ENHANCE COVID IDENTIFICATION PROCESS.

## INTRODUCTION : 
One of the primary clinical observations for screening the novel coronavirus is capturing a chest x-ray image. 
While methods such as reverse transcription polymerase chain reaction (RT-PCR) are predominantly used for testing for COVID-19,these methods have certain shortcomings. 
Such techniques require special testing kits which may not be available in many locations.
Additionally, RT-PCR is time-consuming. Using medical images, we propose a quicker, more accessible method of diagnosis.
In most patients, a chest x-ray contains abnormalities, such as consolidation, resulting from COVID-19 viral pneumonia which sometimes may not be true with the latest strain of the virus, 
hence requiring us to further make the decision based on other data like CT scan. 

In this study, we focus on efficiently detecting imaging features using deep convolutional neural networks in a large dataset and classify them into normal , 
pneumonia and others. This will enhance the speed of the process along with reduction of False negative and false positive cases. 
The cases falling under pneumonia can have further tests like CT scan to decide the case more accurately and precisely.  
It is demonstrated that simple models, alongside the  majority of pretrained networks in the literature, focus on irrelevant features for decision-making. 

In this paper, numerous chest x-ray images from various sources are collected, and the largest publicly accessible dataset is prepared.
Finally, using the transfer learning paradigm, the well-known CheXNet model is utilized to develop COVID-CEXNet. 
This powerful model is capable of detecting the novel coronavirus pneumonia based on relevant and meaningful features with precise localization. 
**COVID-CEXNet** is a step towards a fully automated and robust COVID-19 detection system.

## PROBLEM STATEMENT:
Our focus is to build an end to end classification model to identify signs of Covid-19 from x-rays. 
This will result in the development of a low cost solution to rapidly detect cases of the virus in environments without high end testing infrastructure.
In this research we consolidate the data from multiple Covid detection models.
Additionally, we also combine pre-trained neural networks to effectively discern covid markers from x rays. 

## DATASET : 
The novelty of the disease results in a dearth of data availability.
On consolidating multiple publicly accessible datasets, we were able to create a dataset large enough to facilitate image recognition to a reliable degree. 
Additionally, by using transfer learning, we integrated the classification ability of previously trained detection models by using transfer learning. 
The COVID-CEXNet model was used in this regard.  In it’s training, 780 images of COVID-19 patients were collected in different sizes and formats. 
All collected images are publicly accessible. The dataset includes 5,000 normal CXRs as well as 4,600 images of patients with CAP collected from the NIH CXR-14 dataset.

## DATA PREPROCESSING : 
The limited size of the dataset results in a reduction of accuracy while classifying the chest x-rays due to overfitting of the images. 
To overcome this limitation image augmentation techniques were employed, primarily image rotation. 
The collected images were rotated 90 degrees at a time and were added to the final dataset of images. 
In this way, the diversity of the data available for the training model was significantly increased. 
In addition to image augmentation, we also use image enhancement to increase the contrast within the x-rays to effectively distinguish the non-linearities.
Two adaptive histogram equalization methods were applied namely Contrast Limited Adaptive Histogram Equalization (CLAHE) and  Bi-Histogram Equalization with adaptive sigmoid functions (BEASF). 
CLAHE is an adaptive method that computes several histograms, each corresponding to a distinct section of the image, and uses them to redistribute the lightness values of the image. 
It is therefore suitable for improving the local contrast and enhancing the definitions of edges in each region of an image. 
The CLAHE algorithm is concatenated with BEASF in our method to detect nodular shaped opacities. 
BEASF compliments the CLAHE model to yield images with well defined contrast. 
We pass the augmented images through these enhancement filters before using them to train our neural network. 

## Methodology :
Transfer learning is to benefit from a pretrained model in a new classification task. 
Some pretrained models are trained on millions of images for many epochs and achieved high accuracy on a general task. 
We researched different pretrained models like Resnet and Densenet.
It is worth mentioning that pretrained models are used for fine-tuning, training on target datasets for a small number of epochs, 
instead of retraining for many epochs. Since ImageNet images and labels are different from the CXR dataset, a pretrained model on the same data type should also be considered.
CheXNet is trained on CXR-14, a large publicly available CXR dataset with 14 different diseases such as pneumonia and edema [40]. 
CheXNet claims to have a radiologist-level diagnosis accuracy, has better performance than previous related research studies [33], [41], 
and has simpler architecture than later approaches [42]. CheXNet is based on DenseNet architecture and has been trained on frontal CXRs. 
It also could be used as a better option for the final model backbone. 
The proposed CHNet is a CheXNet-based model, fine-tuned on the dataset with 431 layers and ≈ 7M parameters. 
Our proposed network uses the DenseNet-121 as its backbone. 
There are different architectures utilized to build CheXNet, out of which the DenseNet has shown better capabilities for detecting pulmonary diseases [40]. 
COVID- CEXNet has a FC layer consisting of 10 nodes followed by a dropout layer with a 0.2 dropping rate to prevent overfitting. 
The activation function of the last layer is changed from SoftMax to Sigmoid because the task is a binary classification. 
Our proposed model focused on the higher level classification of the type of infection in lungs in order to speed up the process of detection. 
Along with this CEXNet is benefiting from a lung segmentation module and different image enhancement algorithms within the preprocessing section. 
Moreover, fine-tuning on a larger dataset along with several overfitting-prevention methods such as dropout layer and 
label smoothing will help our model outperform in terms of correctly localizing pneumonia in CXRs.

## MODELS PERFORMANCE : 
The model was constructed using transfer learning and then re-trained with the augmented dataset. 
It classified the images into 3 categories namely, normal, pneumonia and other. Using these classifications we can determine which scans show signs of infection by COVID-19 and which do not.
Given below is the confusion matrix showing the results of classification for our test dataset. 
An accuracy of 96 percent was achieved by correctly classifying 289 images out of a set of 301. 
This demonstrates the reliability of our model, thus validating the proposal for the model to be used as an efficient tool for COVID detection. 
Select images can then be re-examined by a physician to obtain a highly reliable and rapid reading. 

## OBSERVATION AND FUTURE WORK :
In this report, we firstly collected a dataset of CXR images from normal lungs and pneumonia infected patients(which include covid cases) and other lung related diseases.
The constructed dataset is made from images of different datasets from multiple hospitals and radiologists and is the largest public dataset to the best of our knowledge. 
Next, we designed a DenseNet-based model and fine-tuned it with weights initially set from the CheXNet model. 
Comparing model visualization over a batch of samples as well as accuracy scores, we denoted the significance of Grad-CAM heatmaps and its priority to be considered the primary model validation metric. 
Finally, we discussed several points like data shortage and the importance of transfer learning for tackling similar tasks. 
A final CP class f-score of 0.94 for binary classification and 0.85 for three-class classification are achieved. 
The proposed model development procedure is visualization-oriented as it is the best method to confirm its generalization as a medical decision support system.

## DATA AND CODE AVAILABILITY :
Source codes and the pre- trained network weights to help accelerate further research studies.
https://drive.google.com/drive/folders/1BH8KQQKVQSDgOQiIQPIonUUL9UXzzWMu?usp=sharing


## WORK DONE :
**Apeksha**
- Literary survey for the project
- Found the model for transfer learning
- Ran and tested the base model
- Segmentation of images
- Interpretation Colab notebook to run online

**Yogya**
- Literary survey for the project
- Got the dataset and the pre trained weights by contacting the paper authors
- Fine tuning and training the model
- Python script codes to run offline

**Anish**
- Understanding the existing model and worked on Augmentation
- Automating the Dataset reading
- Preparing the Report

**Maha**
- Understanding the existing model
- Preparing the dataset
- Preparing the ReadmeFile

## Learning Outcomes:
- End to end development of software to address real world problems on using open source consolidated data.
- Practical application of transfer learning, image preprocessing and neural network optimization for multi-class classification.
- Tuned programming skills and familiarized with use of python libraries for image processing, machine learning and data visualization.
- Learned from experience of medical professionals directly for a real world study.

## Challenges Faced:
- Limited availability of clean, reliable data.
- Circumscribed by the processing ability of commodity hardware that had been used for classification and analysis.
- Lack of knowledge on medical imaging and virus detection with radiology.










