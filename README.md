# Predicting the Sale Price of Bulldozers using Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-latest-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/status-completed-success.svg)]()

This repository contains an end-to-end machine learning project aimed at predicting the future sale prices of bulldozers based on their historical characteristics and previous auction records. The project follows a structured machine learning workflow from exploratory data analysis (EDA) and feature engineering to model training and evaluation.

---

## 📋 Table of Contents
1. [Problem Definition](#1-problem-definition)
2. [Data Overview](#2-data-overview)
3. [Evaluation Metric](#3-evaluation-metric)
4. [Features & Data Dictionary](#4-features--data-dictionary)
5. [Project Workflow](#5-project-workflow)
6. [Getting Started](#6-getting-started)
7. [Results & Model Performance](#7-results--model-performance)

---

## 1. Problem Definition
The primary objective is to answer the following question:
> *"How well can we predict the future sale price of a bulldozer, given its characteristics and previous examples of how much similar bulldozers have been sold for?"*

This is framed as a **supervised regression problem** where we predict a continuous target variable (`SalePrice`).

---

## 2. Data
The data for this project is sourced from the Kaggle **[Bluebook for Bulldozers](https://www.kaggle.com/c/bluebook-for-bulldozers/data)** competition. It consists of three main datasets:

* **`Train.csv`**: Training set containing historical data through the end of 2011.
* **`Valid.csv`**: Validation set containing data from January 1, 2012, to April 30, 2012. Used for tracking progress and tuning hyperparameters.
* **`Test.csv`**: Test set containing data from May 1, 2012, to November 2012. Used to determine the final ranking/score.

---

## 3. Evaluation
The evaluation metric specified for this competition is **RMSLE** (Root Mean Squared Log Error) between the actual and predicted auction prices. 

Minimizing RMSLE ensures that penalizing under-predictions and over-predictions is treated relative to logarithmic scales. For more details, visit the [Kaggle Evaluation Page](https://www.kaggle.com/c/bluebook-for-bulldozers/overview/evaluation).

---

## 4. Features & Data Dictionary
The dataset contains over 50 columns comprising numerical, categorical, and datetime attributes. 
* A comprehensive breakdown of all features can be found in the official [Google Sheets Data Dictionary](https://docs.google.com/spreadsheets/d/18ly-bLR8sbDJLITkWG7ozKm8l3RyieQ2Fpgix-beSYI/edit?usp=sharing).

---

## 5. Project Workflow
1. **Exploratory Data Analysis (EDA):** Inspecting dataset size, data types, and missing values (`df.info()`, `df.isna().sum()`).
2. **Parsing Dates:** Converting datetime objects (`saledate`) into rich temporal components (Year, Month, Day, Day of Week, Day of Year) to help tree-based models capture seasonal trends.
3. **Handling Missing & Categorical Data:** 
   - Converting string/object types into pandas categorical codes.
   - Filling missing numerical and categorical values.
4. **Model Building & Experimentation:** Training baseline regressors (e.g., Random Forest Regressor) and evaluating performance against validation splits.

---

## 6. Getting Started

### Prerequisites
Make sure you have Python installed along with the required libraries:
```bash
pip install numpy pandas matplotlib scikit-learn
