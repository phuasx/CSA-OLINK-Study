# Predicting Antipsychotic Responsiveness in Schizophrenia Using Machine Learning

This repository contains the code used in the study titled **"Predicting Antipsychotic Responsiveness Using a Machine Learning Classifier Trained on Plasma Levels of Inflammatory Markers in Schizophrenia"**. The code is structured to facilitate feature selection, model building, and explainable AI analysis to predict pharmacological subtypes within schizophrenia, including antipsychotic response, clozapine response, and schizophrenia status.

## Authors

- Jie Yin Yee<sup>1</sup>
- Ser-Xian Phua<sup>2</sup> (GitHub Maintainer)
  - Email: phuasx@gmail.com
- Yuen Mei See<sup>1</sup>
- Anand Kumar Andiappan<sup>3</sup>
- Wilson Wen Bin Goh<sup>2,4,5,6,7</sup>
- Jimmy Lee<sup>1,2</sup>

**Affiliations:**

<sup>1</sup> North Region, Institute of Mental Health, Singapore  
<sup>2</sup> Lee Kong Chian School of Medicine, Nanyang Technological University, Singapore  
<sup>3</sup> Singapore Immunology Network (SIgN), Agency for Science, Technology and Research (A*STAR), Singapore  
<sup>4</sup> Center for Biomedical Informatics, Nanyang Technological University, Singapore  
<sup>5</sup> School of Biological Sciences, Nanyang Technological University, Singapore  
<sup>6</sup> Center of AI in Medicine, Nanyang Technological University, Singapore  
<sup>7</sup> Division of Neurology, Department of Brain Sciences, Faculty of Medicine, Imperial College London

## Repository Structure

This repository contains three main Python scripts:

1. **Feature Selection (SVM RFE workflow)**  
   `feature_selection_olink.py`  
   This script implements Support Vector Machine Recursive Feature Elimination (SVM-RFE) for feature selection. It helps identify the most important inflammatory markers contributing to antipsychotic responsiveness in schizophrenia.

2. **Prediction Modeling**  
   `modelling_olink.py`  
   This script builds the prediction models using the selected features. Various machine learning classifiers are trained and evaluated for their ability to predict pharmacological subtypes, including antipsychotic and clozapine response.

3. **Explainable AI (SHAP analysis)**  
   `shap_xai_olink.py`  
   This script uses SHAP (SHapley Additive exPlanations) to provide explainable insights into the prediction models. SHAP helps visualize the contribution of each feature to the model’s predictions, enhancing interpretability.

## Usage

Each script can be executed independently. Ensure that the input data (plasma levels of inflammatory markers) is correctly formatted as expected by the scripts. Modify the paths and any configuration settings as needed.

## Availability

The data is currently available upon request. Please contact me **Ser-Xian Phua** (phuasx@gmail.com) or any of the authors for access. 

## Citation

If you use this code in your research, please cite the related manuscript:

**"Predicting Antipsychotic Responsiveness Using a Machine Learning Classifier Trained on Plasma Levels of Inflammatory Markers in Schizophrenia"**
