import tensorflow as tf
import re
import numpy as np
import pickle
import cyrtranslit
import sys

class Tokenizer:
  entity_to_word = {
    0: "_URL_",
    1: "_MEMBER_MENTION_",
    2: "_MEMBER_MENTION_",
    3: "_CHANNEL_MENTION_",
    4: "_ROLE_MENTION_",
    5: "_CUSTOM_EMOJI_",
    6: "_ANIMATED_CUSTOM_EMOJI_",
  }

  def __init__(self):
    self.index_to_word = {}
    self.word_to_index = {}

  def fill_index_to_word(self):
    self.index_to_word = {value: key for key, value in self.word_to_index.items()}

  def load_vocab_from_file(self, fname):
    with open(fname, "rb") as file:
      self.word_to_index = pickle.load(file)
      self.fill_index_to_word()

  @staticmethod
  def trigramize(word):
    trigrams = []
    for idx, char in enumerate(word):
      if idx == 0:
        first = "*"
      else:
        first = word[idx - 1]
      second = char
      if idx == len(word) - 1:
        third = "*"
      else:
        third = word[idx + 1]
      trigrams.append(first + second + third)
    return trigrams

  def tokenize(self, content):
    tuples = re.findall(
      r"(http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+)|"
      r"(<@!?\d{16,20}>)|"
      r"(<#\d{16,20}>)|"
      r"(<@&\d{16,20}>)|"
      r"(<:\w{1,32}:\d{16,20}>)|"
      r"(<[a]:\w{1,32}:\d{16,20}>)|"
      r"(@everyone|@here)|"
      r"([^\d\W]+)|"
      r"(.)",
      content,
      re.UNICODE,
    )
    result = []
    result.append("_NOT_MY_MESSAGE_BEGIN_")
    for tup in tuples:
      for idx, item in enumerate(tup):
        if item:
          if idx <= 6:
            result.append(self.entity_to_word[idx])
          elif idx in [7, 8]:
            if item.isupper():
              result.append("_CAPS_")
            elif item[0].isupper():
              result.append("_SHIFT_")
            trigrams = self.trigramize(cyrtranslit.to_cyrillic(item.lower(), "ru"))
            result.extend(trigrams)
          else:
            result.append(item)
    result.append("_MESSAGE_END_")
    return result

  def get_index_by_word(self, word):
    if word in self.word_to_index:
      return self.word_to_index[word]
    else:
      return self.word_to_index["_UNK_"]

  def encode_input(self, message, max_len=64):
    encoder_input_data = np.zeros((1, max_len), dtype="uint16")
    tokenized_input = []
    tokenized_input.extend(self.tokenize(message))
    if len(tokenized_input) > max_len:
      tokenized_input = tokenized_input[-max_len:]
    for idx, token in enumerate(tokenized_input):
      encoder_input_data[0, idx] = self.get_index_by_word(token)
    return encoder_input_data.tolist()[0]

tokenizer = Tokenizer()
tokenizer.load_vocab_from_file("vocab.pickle")

MAX_LENGTH = 64

def load_and_tokenize():
  f = open(sys.argv[1], encoding="utf8") # 12гб ОЗУ каждые 7млн строк (~180 мб)

  inputs = []
  outputs = []
  l1 = None
  i = 0
  while True:
    if i % 100000 == 0:
      print(i)
    if not l1:
      l1 = f.readline().replace("\n", "")
    l2 = f.readline().replace("\n", "")
    if not l2:
      break
    inputs.append(tokenizer.encode_input(l1[:MAX_LENGTH].lower()))
    outputs.append(tokenizer.encode_input(l2[:MAX_LENGTH].lower()))
    l1 = l2
    i += 1

  f.close()

  inputs = tf.keras.preprocessing.sequence.pad_sequences(inputs, maxlen=MAX_LENGTH, padding='post')
  outputs = tf.keras.preprocessing.sequence.pad_sequences(outputs, maxlen=MAX_LENGTH, padding='post')
  return inputs, outputs

questions, answers = load_and_tokenize()
print('Number of samples: {}'.format(len(questions)))

BATCH_SIZE = 64
BUFFER_SIZE = 20000

dataset = tf.data.Dataset.from_tensor_slices(({
  'inputs': questions,
  'dec_inputs': answers[:, :-1]
}, {
  'outputs': answers[:, 1:]
}))

dataset = dataset.cache()
dataset = dataset.shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE)
dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

def scaled_dot_product_attention(query, key, value, mask):
  matmul_qk = tf.matmul(query, key, transpose_b=True)
  depth = tf.cast(tf.shape(key)[-1], tf.float32)
  logits = matmul_qk / tf.math.sqrt(depth)
  if mask is not None:
    logits += (mask * -1e9)

  attention_weights = tf.nn.softmax(logits, axis=-1)
  output = tf.matmul(attention_weights, value)
  return output

class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, name="multi_head_attention"):
    super(MultiHeadAttention, self).__init__(name=name)
    self.num_heads = num_heads
    self.d_model = d_model
    assert d_model % self.num_heads == 0
    self.depth = d_model // self.num_heads
    self.query_dense = tf.keras.layers.Dense(units=d_model)
    self.key_dense = tf.keras.layers.Dense(units=d_model)
    self.value_dense = tf.keras.layers.Dense(units=d_model)
    self.dense = tf.keras.layers.Dense(units=d_model)

  def split_heads(self, inputs, batch_size):
    inputs = tf.reshape(inputs, shape=(batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(inputs, perm=[0, 2, 1, 3])

  def call(self, inputs):
    query, key, value, mask = inputs['query'], inputs['key'], inputs['value'], inputs['mask']
    batch_size = tf.shape(query)[0]
    query = self.query_dense(query)
    key = self.key_dense(key)
    value = self.value_dense(value)
    query = self.split_heads(query, batch_size)
    key = self.split_heads(key, batch_size)
    value = self.split_heads(value, batch_size)
    scaled_attention = scaled_dot_product_attention(query, key, value, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
    concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

    outputs = self.dense(concat_attention)
    return outputs

  def get_config(self):
    config = super().get_config()
    config.update({
        "num_heads": self.num_heads,
        "d_model": self.d_model
    })
    return config

def create_padding_mask(x):
  mask = tf.cast(tf.math.equal(x, 0), tf.float32)
  return mask[:, tf.newaxis, tf.newaxis, :]

def create_look_ahead_mask(x):
  seq_len = tf.shape(x)[1]
  look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
  padding_mask = create_padding_mask(x)
  return tf.maximum(look_ahead_mask, padding_mask)

class PositionalEncoding(tf.keras.layers.Layer):
  def __init__(self, position, d_model):
    super(PositionalEncoding, self).__init__()
    self._position = position
    self._d_model = d_model
    self.pos_encoding = self.positional_encoding(position, d_model)

  def get_angles(self, position, i, d_model):
    angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    return position * angles

  def positional_encoding(self, position, d_model):
    angle_rads = self.get_angles(
        position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
        i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
        d_model=d_model
    )

    sines = tf.math.sin(angle_rads[:, 0::2])
    cosines = tf.math.cos(angle_rads[:, 1::2])
    pos_encoding = tf.concat([sines, cosines], axis=-1)
    pos_encoding = pos_encoding[tf.newaxis, ...]
    return tf.cast(pos_encoding, tf.float32)

  def call(self, inputs):
    return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

  def get_config(self):
    config = super().get_config()
    config.update({
        "position": self._position,
        "d_model": self._d_model
    })
    return config

def encoder_layer(units, d_model, num_heads, dropout, name="encoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
  attention = MultiHeadAttention(
      d_model, num_heads, name="attention")({
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': padding_mask
      })

  attention = tf.keras.layers.Dropout(rate=dropout)(attention)
  attention = tf.keras.layers.LayerNormalization(epsilon=1e-6)(inputs + attention)
  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention + outputs)
  return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def encoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name="encoder"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name="padding_mask")
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)

  for i in range(num_layers):
    outputs = encoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name="encoder_layer_{}".format(i),
    )([outputs, padding_mask])

  return tf.keras.Model(inputs=[inputs, padding_mask], outputs=outputs, name=name)

def decoder_layer(units, d_model, num_heads, dropout, name="decoder_layer"):
  inputs = tf.keras.Input(shape=(None, d_model), name="inputs")
  enc_outputs = tf.keras.Input(shape=(None, d_model), name="encoder_outputs")
  look_ahead_mask = tf.keras.Input(shape=(1, None, None), name="look_ahead_mask")
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  attention1 = MultiHeadAttention(
      d_model, num_heads, name="attention_1")(inputs={
          'query': inputs,
          'key': inputs,
          'value': inputs,
          'mask': look_ahead_mask
      })

  attention1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention1 + inputs)
  attention2 = MultiHeadAttention(
      d_model, num_heads, name="attention_2")(inputs={
          'query': attention1,
          'key': enc_outputs,
          'value': enc_outputs,
          'mask': padding_mask
      })

  attention2 = tf.keras.layers.Dropout(rate=dropout)(attention2)
  attention2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(attention2 + attention1)

  outputs = tf.keras.layers.Dense(units=units, activation='relu')(attention2)
  outputs = tf.keras.layers.Dense(units=d_model)(outputs)
  outputs = tf.keras.layers.Dropout(rate=dropout)(outputs)
  outputs = tf.keras.layers.LayerNormalization(epsilon=1e-6)(outputs + attention2)

  return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

def decoder(vocab_size, num_layers, units, d_model, num_heads, dropout, name='decoder'):
  inputs = tf.keras.Input(shape=(None,), name='inputs')
  enc_outputs = tf.keras.Input(shape=(None, d_model), name='encoder_outputs')
  look_ahead_mask = tf.keras.Input(shape=(1, None, None), name='look_ahead_mask')
  padding_mask = tf.keras.Input(shape=(1, 1, None), name='padding_mask')
  embeddings = tf.keras.layers.Embedding(vocab_size, d_model)(inputs)
  embeddings *= tf.math.sqrt(tf.cast(d_model, tf.float32))
  embeddings = PositionalEncoding(vocab_size, d_model)(embeddings)
  outputs = tf.keras.layers.Dropout(rate=dropout)(embeddings)
  for i in range(num_layers):
    outputs = decoder_layer(
        units=units,
        d_model=d_model,
        num_heads=num_heads,
        dropout=dropout,
        name='decoder_layer_{}'.format(i),
    )(inputs=[outputs, enc_outputs, look_ahead_mask, padding_mask])

  return tf.keras.Model(inputs=[inputs, enc_outputs, look_ahead_mask, padding_mask], outputs=outputs, name=name)

def transformer(vocab_size, num_layers, units, d_model, num_heads, dropout, name="transformer"):
  inputs = tf.keras.Input(shape=(None,), name="inputs")
  dec_inputs = tf.keras.Input(shape=(None,), name="dec_inputs")

  enc_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='enc_padding_mask')(inputs)
  look_ahead_mask = tf.keras.layers.Lambda(create_look_ahead_mask, output_shape=(1, None, None), name='look_ahead_mask')(dec_inputs)
  dec_padding_mask = tf.keras.layers.Lambda(create_padding_mask, output_shape=(1, 1, None), name='dec_padding_mask')(inputs)
  enc_outputs = encoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[inputs, enc_padding_mask])

  dec_outputs = decoder(
      vocab_size=vocab_size,
      num_layers=num_layers,
      units=units,
      d_model=d_model,
      num_heads=num_heads,
      dropout=dropout,
  )(inputs=[dec_inputs, enc_outputs, look_ahead_mask, dec_padding_mask])

  outputs = tf.keras.layers.Dense(units=vocab_size, name="outputs")(dec_outputs)
  return tf.keras.Model(inputs=[inputs, dec_inputs], outputs=outputs, name=name)

tf.keras.backend.clear_session()

NUM_LAYERS = 6
D_MODEL = 512
NUM_HEADS = 8
UNITS = 2048
DROPOUT = 0.1
VOCAB_SIZE = 10000
model = transformer(
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
    units=UNITS,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    dropout=DROPOUT
)

from os import path as opath
if opath.exists("test_weights.h5"):
  print("Loading weights...")
  model.load_weights("test_weights.h5")

def loss_function(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')(y_true, y_pred)
  mask = tf.cast(tf.not_equal(y_true, 0), tf.float32)
  loss = tf.multiply(loss, mask)
  return tf.reduce_mean(loss)

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()
    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)
    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps**-1.5)
    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

  def get_config(self):
    config = {
        "warmup_steps": self.warmup_steps,
        "d_model": self.d_model
    }
    return config

learning_rate = CustomSchedule(D_MODEL)
optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)

def accuracy(y_true, y_pred):
  y_true = tf.reshape(y_true, shape=(-1, MAX_LENGTH - 1))
  return tf.keras.metrics.sparse_categorical_accuracy(y_true, y_pred)

model.compile(optimizer=optimizer, loss=loss_function, metrics=[accuracy])
EPOCHS = 3
try:
  model.fit(dataset, epochs=EPOCHS)
except KeyboardInterrupt:
  model.save_weights("test_weights.h5")
  exit()
model.save_weights("test_weights.h5")
