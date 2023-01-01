import tensorflow as tf

import numpy as np
import os


# config
BATCH_SIZE = 64
BUFFER_SIZE = 10000
SEQ_LENGTH = 100
EMBED_DIM = 256
RNN_UNITS = 1024
EPOCHS=30


# load dataset
data_url = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

dataset_text = open(data_url, 'rb').read().decode(encoding='utf-8')



# map text to numbers
## obtain the unique characters in the dataset
vocab = sorted(set(dataset_text))
## create a mapping from unique characters to indices
char2idx = {char:index for index, char in enumerate(vocab)}
idx2char = np.array(vocab)
# convert dataset from 'characters' to 'integers'
text_as_int = np.array([char2idx[char] for char in dataset_text])


# breaking up data into sequences
## calculate number of examples per epoch for sequence length of 100 characters 
examples_per_epoch = len(dataset_text)//SEQ_LENGTH
## the dataset holds around 1 mio. characters
## so this generates around 10k sequences
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
sequences = char_dataset.batch(SEQ_LENGTH+1, drop_remainder=True)

## duplicate each sequence and shift +1/-1 it to form the input and target text:
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

# ## visualize dataset
# for input_example, target_example in  dataset.take(2):
#   print ('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
#   print ('Target data:', repr(''.join(idx2char[target_example.numpy()])))

## shuffle the dataset and create batches
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# model building
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
  batch_size=BATCH_SIZE)


for input_example_batch, target_example_batch in dataset.take(10):
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")


## loss function
def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss  = loss(target_example_batch, example_batch_predictions)

model.compile(optimizer='adam', loss=loss)


# directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# model training
history = model.fit(dataset, epochs=EPOCHS, callbacks=[checkpoint_callback])