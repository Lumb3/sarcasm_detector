import json
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_token = '<OOV>' # If the token is unrecognized
training_size = 20000

# Load sarcasm dataset from JSON file (one JSON object per line)
data = []
with open("Data/sarcasm.json") as json_file:
    for line in json_file:
        data.append(json.loads(line))

sentences = []
labels = []
for item in data:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])

# Splitting data into training and testing sets
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]
training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# Create tokenizer with vocab limit and OOV token for unknown words
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
tokenizer.fit_on_texts(training_sentences)  # Learn word indices from training data

word_index = tokenizer.word_index  # Dictionary of word to integer mapping

# Convert sentences to sequences of integers
training_sequences = tokenizer.texts_to_sequences(training_sentences)
# Pad sequences to uniform length for model input
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Turn the data into numpy arrays (required for TensorFlow)
training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_labels = np.array(testing_labels) # The correct answers
testing_padded = np.array(testing_padded) # Data to train

# Creating a neural network model with layers that learns patterns from data
model = tf.keras.Sequential([
    # num of unique words, size of each word vector
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length), # Turns words (tokens) into dense vectors
    tf.keras.layers.GlobalAveragePooling1D(), # summarizing the whole sentence into one meaningful vector
    tf.keras.layers.Dense(24, activation='relu'), # 24 tiny detectors looking for features in the sentence.
    tf.keras.layers.Dense(1, activation='sigmoid') # a final yes-or-no decision
])
# track accuracy while training
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

num_epochs = 30 # number of passes, as epoches increase, model learns more (the model will go through the entire training dataset 30 times)

# Training the model (self-learning)
# means you'll see 1 line per epoch
history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)
model.evaluate(testing_padded, testing_labels) # Comparing the trained data with the correct answers (testing_labels)


# Testing
test_sentences = [
    "granny starting to fear spiders in the garden might be real haha crazy",
    "game of thrones season finale showing this sunday night"
]

# Convert sentences to sequences and pad them
test_sequences = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# Get predictions (model outputs probabilities of sarcasm)
predictions = model.predict(test_padded)

# Display results
for i, sentence in enumerate(test_sentences):
    probability = predictions[i][0]
    percentage = probability * 100
    print(f"Sentence: \"{sentence}\"")
    print(f"Predicted Sarcasm Probability: {percentage:.2f}%\n")


# Save model
model.save("sarcasm_model.keras")

# Save tokenizer
import pickle
with open("tokenizer.pickle", "wb") as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
