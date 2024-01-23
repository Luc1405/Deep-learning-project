import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
#Summarize the history
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
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

#Use the model to predict your own messages. Maximum of three (can always be changed)
text_messages = []
max_length_list = 3

while len(text_messages) < max_length_list:
    item = input("Enter your Item to the List: ")
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