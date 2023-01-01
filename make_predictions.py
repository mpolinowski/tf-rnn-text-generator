import tensorflow as tf

import numpy as np
import os


# config
EMBED_DIM = 256
RNN_UNITS = 1024


# get vocabulary
data_url = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
dataset_text = open(data_url, 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(dataset_text))
char2idx = {char:index for index, char in enumerate(vocab)}
idx2char = np.array(vocab)


# rebuild model for smaller batch size (of `1`)
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])

  return model


model = build_model(
  vocab_size = len(vocab),
  embedding_dim=EMBED_DIM,
  rnn_units=RNN_UNITS,
  batch_size=1)


## locate latest checkpoint
checkpoint_dir = './training_checkpoints'
tf.train.latest_checkpoint(checkpoint_dir)


## load weights from checkpoint
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir)).expect_partial()
model.build(tf.TensorShape([1, None]))


# prediction loop
def generate_text(model, start_string):
  # evaluation step (generating text using the learned model)

  # number of characters to generate
  num_generate = 1000

  # converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # low lsd results in more predictable text.
  # higher lsd results in more surprising text.
  # experiment to find the best setting.
  lsd = 0.3

  # here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the word returned by the model
      predictions = predictions / lsd
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


print(generate_text(model, start_string=u"KATHERINE: "))