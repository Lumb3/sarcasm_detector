# 🤖 Sarcasm Detection Using Tokenization and Neural Networks

This project builds a sarcasm detection system using a neural network trained on real news headlines. It processes text by turning words into numbers and feeds them into a model that improves itself through learning.

---

## 📊 Dataset

- **Source**: [Kaggle - News Headlines Dataset for Sarcasm Detection](https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection)
- **Format**: JSON (each line = one JSON object)
- **Fields**:
  - `headline`: the text of the news headline
  - `is_sarcastic`: label (1 for sarcastic, 0 for not sarcastic)

---

## 🧪 How It Works (Step-by-Step)

### 1. Load and Prepare Data
- Headlines and sarcasm labels are loaded from a JSON file.
- First 20,000 examples are used for training, and the rest for testing.

### 2. Tokenization: Text → Numbers
- A Keras `Tokenizer` is created with a vocabulary size limit of 10,000.
- Words in the training data are mapped to unique integers (tokens).
- Words not in the vocabulary are replaced with `<OOV>` (out-of-vocabulary).

### 3. Padding Sequences
- Headline sequences are converted to the same length (100) using **post-padding** and **post-truncation**.
- This makes the data uniform in size, which is required for input to the neural network.

### 4. Convert to Numpy Arrays
- Both padded sequences and labels are converted into NumPy arrays.
- This step is necessary because TensorFlow models require numerical array input.

### 5. Build Neural Network Model
The model is created using `tf.keras.Sequential()` with the following layers:
- `Embedding`: Converts each word index into a dense 16-dimensional vector.
- `GlobalAveragePooling1D`: Reduces the sequence into a single feature vector.
- `Dense (24, relu)`: Learns hidden patterns.
- `Dense (1, sigmoid)`: Outputs a probability score between 0 and 1 for sarcasm.

### 6. Train the Model
- The model is compiled using binary cross-entropy loss and the Adam optimizer.
- It is trained for 30 epochs.
- During training, the model evaluates itself on the test set after each epoch to improve performance.

### 7. Evaluate and Test
- After training, the model's accuracy is measured on the testing dataset.
- You can input new headlines to see the model predict sarcasm probability.

### 8. Save the Model and Tokenizer
- The trained model is saved as `sarcasm_model.keras`.
- The tokenizer is saved using Python’s `pickle` module as `tokenizer.pickle`.

---

## 🧠 Example Prediction

```python

Input:

test_sentences = [
    "granny starting to fear spiders in the garden might be real haha crazy",
    "game of thrones season finale showing this sunday night"
]

Output:
Sentence: "granny starting to fear spiders in the garden might be real haha crazy"
Predicted Sarcasm Probability: 89.23%

Sentence: "game of thrones season finale showing this sunday night"
Predicted Sarcasm Probability: 13.56%
