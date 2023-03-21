To extract specific information like the sport name, venue, and time from a text, you can follow these steps:

1. Preprocess the text: Clean and preprocess the text to remove any irrelevant characters, punctuation, or unnecessary whitespace. This will help improve the performance of the Named Entity Recognition (NER) and text classification algorithms.

2. Named Entity Recognition: Use an NER model to identify entities like venue and time in the text. Popular NLP libraries like spaCy, NLTK, or Stanford NLP provide pre-trained NER models that can be used for this purpose.

3. Train a text classifier: Train a text classification model to classify the sport name. You can use various algorithms such as Naive Bayes, Support Vector Machines, or deep learning models like Convolutional Neural Networks (CNN) or Recurrent Neural Networks (RNN). Popular libraries for text classification include scikit-learn, TensorFlow, Keras, and PyTorch.

4. Combine the results: After extracting the venue and time using NER and predicting the sport name using the text classifier, combine the results to generate the final output.

Here's a high-level overview of the steps:

Step 1: Preprocess the text
```
def preprocess(text):
    # Implement your preprocessing logic
    return preprocessed_text
```
Step 2: Extract venue and time using NER

```
def extract_venue_time(text):
    # Implement your NER logic
    return venue, time
```
Step 3: Train a text classifier for sport names

```
def train_sport_classifier(training_data):
    # Train your text classifier
    return model
```

Step 4: Predict the sport name using the text classifier

```
def predict_sport(model, text):
    # Implement your sport prediction logic
    return sport
```

Example usage:

```
preprocessed_text = preprocess(text)
venue, time = extract_venue_time(preprocessed_text)
sport = predict_sport(model, preprocessed_text)
print(f"Sport: {sport}\nVenue: {venue}\nTime: {time}")
```

By following these steps, you can extract the sport name, venue, and time from a given text. Note that the quality of the results will depend on the performance 
of your NER and text classification models, which can be improved by using more training data and fine-tuning the model parameters.
