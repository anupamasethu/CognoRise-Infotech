# TASK 1: IRIS FLOWER CLASSIFICATION

## Objective:
The objective of the document "task-1-iris-flower-classification" is to demonstrate the classification of iris flowers using machine learning techniques, specifically focusing on the Random Forest Classifier. The goal is to accurately predict the class of iris flowers based on their attributes.

## Dataset:
The dataset used in this document is the famous Iris dataset, first introduced by Sir R.A. Fisher. It contains information on four numeric predictive attributes: sepal length, sepal width, petal length, and petal width. There are three classes of iris plants in the dataset: Setosa, Versicolour, and Virginica. The dataset consists of 150 instances, with 50 instances for each class.

## Implementation:
The implementation involves several steps:
1. **Data Exploration:** Exploring the dataset to understand its structure and attributes.
2. **Data Splitting:** Splitting the data into features (X) and target (y).
3. **Data Normalization:** Normalizing the data to a common scale using StandardScaler.
4. **Model Selection:** Choosing the Random Forest Classifier for classification.
5. **Model Training:** Training the Random Forest Classifier on the training data.
6. **Model Evaluation:** Evaluating the model's performance using accuracy score, classification report, and confusion matrix.
7. **Prediction:** Making predictions on new samples using the trained model.

## Usage:
The document serves as a guide for implementing iris flower classification using machine learning techniques, specifically the Random Forest Classifier. It provides a step-by-step approach to explore the dataset, train the model, evaluate its performance, and make predictions on new samples. Users can refer to this document to understand how to apply machine learning algorithms for iris flower classification.

## Dependencies:
The implementation in the document relies on several Python libraries and modules:
- numpy
- pandas
- sklearn.datasets
- sklearn.model_selection
- sklearn.linear_model
- sklearn.preprocessing
- sklearn.metrics
- seaborn
- matplotlib.pyplot


# TASK 3: TITANIC SURVIVAL PREDICTION

## Objective:
The objective of the document "task-3-titanic-survival-prediction.pdf" is to predict survival outcomes on the Titanic using machine learning techniques. The goal is to analyze the dataset, preprocess the data, train a model, and make accurate predictions on survival status.

## Dataset:
The dataset used in the document contains information about passengers on the Titanic, including features like PassengerId, Survived, Pclass, Name, Sex, Age, SibSp, Parch, Fare, Cabin, and Embarked. This dataset is crucial for training the machine learning model to predict survival outcomes accurately.

## Implementation:
The implementation involves various steps such as loading the dataset, handling missing values, encoding categorical data (like Sex and Embarked), splitting the data for training and testing, normalizing the data, exploring correlations using a heatmap, and training a Random Forest model for prediction. The document also includes assessing model accuracy and generating predictions for new samples.

## Usage:
The document serves as a guide for individuals interested in data analysis and machine learning. It provides a practical example of how to approach a survival prediction problem using the Titanic dataset. By following the steps outlined in the document, users can learn how to preprocess data, train a machine learning model, and evaluate its performance.

## Dependencies:
The document utilizes several Python libraries and tools for data analysis and machine learning, including:
- pandas 
- numpy 
- sklearn 
- matplotlib 
- seaborn 

# TASK 8: FAKE NEWS PREDICTION

## Objective:
The objective of the document is to predict fake news using a machine learning model. The goal is to classify news articles as either real or fake based on the content.

## Dataset:
The dataset used in the document is stored in a CSV file named 'news.csv'. It contains columns like 'title', 'text', and 'label' where 'label' indicates whether the news is real or fake.

## Approach:
The approach involves several steps:
1. **Data preprocessing:** Lowercasing text, removing non-alphabetic characters, splitting text into words, applying stemming, and filtering out stopwords.
2. **Feature extraction:** Using TF-IDF vectorization to convert text data into numerical features.
3. **Model training:** Splitting the data into training and testing sets, training a Logistic Regression model on the training data.
4. **Model evaluation:** Calculating the accuracy of the model on both the training and testing sets.

## Implementation:
The implementation includes:
- Importing necessary libraries like pandas, numpy, nltk, and sklearn.
- Checking and downloading stopwords using NLTK.
- Loading the dataset from 'news.csv' into a DataFrame.
- Applying the stemming function to the 'text' column.
- Vectorizing the text data using TF-IDF.
- Splitting the data into training and testing sets.
- Training a Logistic Regression model on the training data.
- Evaluating the model's accuracy on both training and testing sets.
- Predicting the label of a specific news article and determining if it's real or fake based on the model's prediction.

## Dependencies:
The implementation in the document relies on several Python libraries and modules:

The dependencies used in the document for predicting fake news include:
1. Pandas
2. NumPy
3. NLTK (Natural Language Toolkit)
4. Regular Expressions (re)
5. TfidfVectorizer from sklearn.feature_extraction.text
6. Train_test_split from sklearn.model_selection
7. LogisticRegression from sklearn.linear_model
8. Accuracy_score from sklearn.metrics
