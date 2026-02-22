from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences



# model = load_model("models/rnn_model.h5")
# (_, _), (x_test, y_test) = imdb.load_data(num_words=10000)
# x_test = pad_sequences(x_test, maxlen=200)
# loss, acc = model.evaluate(x_test, y_test)
# print(f"Test Accuracy: {acc:.4f}")





model = load_model("models/lstm_model.h5")
(_, _), (x_test, y_test) = imdb.load_data(num_words=10000)
x_test = pad_sequences(x_test, maxlen=200)
loss, acc = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {acc:.4f}")