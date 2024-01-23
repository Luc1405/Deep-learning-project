import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import confusion_matrix, classification_report

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io


# Import the dataset
dataset = pd.read_csv('spam.csv')

# Make a list of the catagories and messages for easy usability
sentences = dataset['Message'].tolist()
labels = dataset['Category'].tolist()

# Create a number which decides how big the training size is
training_size = int(len(sentences) * 0.8)

# Turn the dataset into training and testing data
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Tuen the labels into arrays (important for use with numpy later)
training_labels_final = np.array(training_labels)
testing_labels_final = np.array(testing_labels)

#List of variables for later use
vocab_size = 600
embedding_dim = 16
max_length = 60
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

#Create the tokenizer
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(training_sentences)

#Pad sequences so that they can be used correctly by the model
padded = pad_sequences(sequences,maxlen=max_length, padding=padding_type, 
                       truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences,maxlen=max_length, 
                               padding=padding_type, truncating=trunc_type)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_review(padded[1]))
print(training_sentences[1])

# Note the embedding layer is first, and the output is only 1 node as it is either 0 or 1
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(6, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.summary()

num_epochs = 30
history=model.fit(padded, training_labels_final, epochs=num_epochs, validation_data=(testing_padded, testing_labels_final))

#List all the data in the history
print(history.history.keys())

# Set the seaborn theme for nice plot styles
sns.set_theme(style="whitegrid")

# Set the figure size for better visibility
plt.figure(figsize=(12, 5))

# Plot for accuracy
plt.subplot(1, 2, 1) # 1 row, 2 columns, first subplot
plt.plot(history.history['accuracy'], label='Train Accuracy', color='blue', linestyle='dashed')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange', linestyle='solid')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='lower right')
plt.ylim([0, 1])  # Set y-axis limits for better scale

# Plot for loss
plt.subplot(1, 2, 2) # 1 row, 2 columns, second subplot
plt.plot(history.history['loss'], label='Train Loss', color='green', linestyle='dashed')
plt.plot(history.history['val_loss'], label='Validation Loss', color='red', linestyle='solid')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper right')
plt.ylim([0, 1])  # Set y-axis limits for better scale

plt.tight_layout()
plt.show()

#Retrieve the weights of the embedding (e) layer
e = model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

#Write out embedding vectors and metadata
out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close() 

# Function to plot the distribution of correct and incorrect predictions
def plot_prediction_distribution(predictions, true_labels, dataset_name='Training'):
    # Convert predictions to label indices (0 or 1)
    predicted_labels = [1 if x > 0.5 else 0 for x in predictions.flatten()]

    # Calculate correct and incorrect predictions
    correct_predictions = np.equal(predicted_labels, true_labels).astype(int)
    correct_counts = np.bincount(correct_predictions, minlength=2)
    
    # Define the colors for the bar chart
    colors = ['red' if x == 0 else 'green' for x in correct_predictions]
    
    # Create a bar chart
    sns.barplot(x=['Incorrect', 'Correct'], y=correct_counts, palette=['red', 'green'])
    plt.title(f'Distribution of Predictions for {dataset_name} Set')
    plt.xlabel('Prediction Type')
    plt.ylabel('Count')
    plt.show()

# Function to calculate and print the confusion matrix and classification report
# and print out incorrectly classified texts
def evaluate_model(predictions, true_labels, sentences, dataset_name='Training'):
    # Convert predictions to label indices (0 or 1)
    predicted_labels = [1 if x > 0.5 else 0 for x in predictions.flatten()]
    
    # Generate the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)
    
    # Convert the confusion matrix into a DataFrame
    cm_df = pd.DataFrame(cm, 
                         index = ['Ham (Actual)', 'Spam (Actual)'], 
                         columns = ['Ham (Predicted)', 'Spam (Predicted)'])
    
    # Display the confusion matrix as a DataFrame
    print(f"{dataset_name} Confusion Matrix:")
    print(cm_df)
    
    # Generate and print the classification report
    print(f"\n{dataset_name} Classification Report:")
    report = classification_report(true_labels, predicted_labels, target_names=['Ham', 'Spam'], output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    print(report_df)
    print("\n")
    
    # Print out incorrectly classified texts
    incorrects = np.where(predicted_labels != true_labels)[0]
    print(f"Incorrectly Classified Texts from {dataset_name} Set:")
    for index in incorrects:
        print(f"Predicted: {'Spam' if predicted_labels[index] else 'Ham'} - Actual: {'Spam' if true_labels[index] else 'Ham'} - Text: {sentences[index]}")
    print("\n")

# Evaluate the model and plot the distribution for the training set
train_predictions = model.predict(padded)
evaluate_model(train_predictions, training_labels_final, training_sentences, dataset_name='Training')
plot_prediction_distribution(train_predictions, training_labels_final, dataset_name='Training')

# Evaluate the model and plot the distribution for the validation set
validation_predictions = model.predict(testing_padded)
evaluate_model(validation_predictions, testing_labels_final, testing_sentences, dataset_name='Validation')
plot_prediction_distribution(validation_predictions, testing_labels_final, dataset_name='Validation')

#Use the model to predict your own messages. Maximum of three (can always be changed)
text_messages = []
max_length_list = 3

while len(text_messages) < max_length_list:
    item = input("Add your own message to predict: ")
    text_messages.append(item)
    print(text_messages)

# Create the sequences
padding_type='post'
sample_sequences = tokenizer.texts_to_sequences(text_messages)
fakes_padded = pad_sequences(sample_sequences, padding=padding_type, maxlen=max_length)           

classes = model.predict(fakes_padded)

# Class ranges from 0 to 1, the closer to 1 the more likely it is to be spam.
for x in range(len(text_messages)):
  print(text_messages[x])
  print(classes[x])
  print('\n')

#This can be uncommented if the model is to be used later and needs to be saved
# saved_model = model.save('my_model.keras')