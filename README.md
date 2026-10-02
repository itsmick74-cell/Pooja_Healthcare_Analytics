# Healthcare Analytics for Doctor Visits

AICTE TIRTC DIY Project

**Student:** Pooja
**College:** Dronacharya Government College
**AICTE STU ID:** STU6a6c52cf9bb8d1785483983

## About
Analysis of a healthcare survey dataset (5,190 records, 12 columns) to find which factors are linked to the number of doctor visits.

## What is done
- Data cleaning and encoding (age scaled to years, yes/no columns to 0/1)
- Exploratory analysis with charts and a correlation heatmap
- Poisson regression for visit counts
- Random Forest and Logistic Regression to predict whether a person visits a doctor

## Key results
- Only 20.2% of people had at least one doctor visit
- Reduced-activity days (r = 0.42) and number of illnesses (r = 0.22) are the strongest links with visits
- Each extra illness raises expected visits by about 20% (Poisson rate ratio 1.20)
- Random Forest: 73.8% accuracy, ROC-AUC 0.737

## Files
- `healthcare_analysis.ipynb`: full notebook with outputs
- `healthcare_analysis.py`: same code as a script
- `1776250375-P2-Healthcare_Analytics_for_Doctor_Visits__1_.csv`: dataset
- `requirements.txt`: Python libraries

## How to run
```
pip install -r requirements.txt
jupyter notebook healthcare_analysis.ipynb
```
Or open the notebook in Google Colab and upload the CSV to the same folder.
