Text Classification with Naive Bayes
This project demonstrates a simple text classification model using Naive Bayes and scikit-learn. The model is designed to classify text data into predefined categories. The dataset is loaded in Parquet format, and the model utilizes a text vectorization technique and Multinomial Naive Bayes for classification.

Requirements

Python 3.10 +

scikit-learn

pandas

numpy

You can install the necessary dependencies using pip:

```
pip install scikit-learn pandas numpy pyarrow

```
Files

train.parquet: Training dataset containing the text and labels for training.

test.parquet: Test dataset for evaluating the model.

habari: Sample input text for classification.

Steps

Loading Data:

The dataset is loaded from Parquet format using ``` pd.read_parquet() ```

The training data (X_train, y_train) and test data (X_test, y_test) are separated.

Text Vectorization:

A CountVectorizer is used to convert the text data into a numerical representation (bag of words).

Alternatively, TfidfVectorizer could be used for better performance on some tasks.

Model Creation:

A Naive Bayes model (MultinomialNB) is used for classification. A pipeline is created to combine the vectorizer and classifier.

Model Training:

The model is trained using the fit() method on the training data.

Prediction:

The trained model is used to predict the labels for the test data.

Evaluation:

Accuracy of the model is calculated by comparing predicted labels with the true labels from the test data.

```
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

# Load data that are in parquet format
train_data = pd.read_parquet('data/train.parquet')
test_data = pd.read_parquet('data/test.parquet')
X_train = train_data['text']
y_train = train_data['label']
X_test = test_data['text']
y_test = test_data['label']

# Create a pipeline with Naïve Bayes
model = make_pipeline(CountVectorizer(), MultinomialNB(alpha=0.8))

# Train the model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```
Sample Prediction
For a sample input text (e.g., habari), the model predicts a category. In this case, the prediction would be:
```

habari = '''
Magonjwa yasiyo ya kuambukiza ni magonjwa ambayo hayasambazwi kwa mtu mwingine kwa kuwa hakuna vimelea mwilini vinavyohusiana na magonjwa hayo...
'''

# Model prediction
model.predict([habari])
Output:


array([5])  # Category: Afya
Categories
0: Uchumi

1: Kitaifa

2: Michezo

3: Kimataifa

4: Burudani

5: Afya
```
Conclusion
This model demonstrates basic text classification using the Naive Bayes algorithm, providing a simple yet effective approach to categorizing text data.
