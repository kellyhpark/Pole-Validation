# CERA: A Community-Driven Platform for Enhancing Utility Safety

This repository aims to replicate the workflow of a potential utility app or platform that will allow users to report faulty electrical poles by taking photos of them and provide a written description of the problem. By training the DETR (End-to-End Object Detection with Transformers) object detection model with a custom dataset, and applying two NLP (Natural Language Processing) models to a sample disaster dataset, we aim to fine-tune these models and provide a proof of concept. Through this, we hope to prove the possibility and uses of a customer report app for utility companies.

## Data Sources
Data for this project is stored in the ```data``` directory in the repo, which contains training and validation images for the ```optimize_detr.ipynb``` notebook, which are manually collected images of poles in the neighborhood.

The ```HumAID``` data, used for ```sentiment_urgency_analysis.ipynb``` and ```topic_classification.ipynb``` is sourced from [CRISIS NLP](https://crisisnlp.qcri.org/humaid_dataset).

## Setup

### Conda Environment
After cloning the repo, navigate to root folder and run:
```
conda env create -f environment.yml
```

### CERA Model Demo
Run the CERA Model Demo page by running the following after setup is complete:
```
streamlit run app.py
```

In addition, the three notebooks available in the ```notebooks``` directory serve the purpose of replicating the visualizations and results of the CERA proof of concept.

### Before Fine-tuning DETR
As fine-tuning cannot be run without a GPU, the ```optimize_detr.ipynb``` notebook **must** be run on [Google Colab](https://colab.google/) or a PC with a GPU. Running the notebook on Google Colab may be done by downloading the finetuning notebook and uploading. If running the notebook on a PC with a GPU, additional steps listed within the notebook.

When prompted to do so within the notebook, please upload the image training and validation data to their respective directories. Additionally locate the custom_train.json and custom_val.json files in the 'annotation' directory and upload/move accordingly as instructed in the notebook.

Please note that running the ```sentiment_urgency_analysis.ipynb``` and ```topic_classification.ipynb``` notebooks do not require a GPU to be run.

## Project Structure

```
├── annotations/
│   ├── custom_train.json
│   ├── custom_val.json
├── data/
│   ├── HumAID/
│   ├── training/
│   ├── validation/
├── models/
│   ├── detr_results/
│   ├── svm_model.pkl
├── notebooks/
│   ├── optimize_detr.ipynb
│   ├── sentiment_urgency_analysis.ipynb
│   ├── topic_classification.ipynb
├── app.py <- demo app 
├── README.md
├── environment.yml
└── .gitignore
```
