import tensorflow as tf
import keras
from transformers import BertTokenizer, TFBertModel

# 1. Load pretrained BERT tokenizer and model
bert_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
bert_encoder = TFBertModel.from_pretrained(bert_model_name)

# 2. Prepare data: example sentences
sentences = [
    "The capital of France is",
    "The largest planet in our solar system is",
    "2 + 2 equals"
]
# 3. Tokenize and build input tensors
inputs = tokenizer(sentences, return_tensors='tf', padding=True)
input_ids = inputs['input_ids']      # shape (batch_size, seq_len)
attention_mask = inputs['attention_mask']

# 4. Define a simple LM head on top of BERT
vocab_size = tokenizer.vocab_size
sequence_length = input_ids.shape[1]

# BERT encoder inputs
ids = keras.Input(shape=(sequence_length,), dtype=tf.int32, name='input_ids')
attn = keras.Input(shape=(sequence_length,), dtype=tf.int32, name='attention_mask')

# Get BERT outputs
bert_outputs = bert_encoder(ids, attention_mask=attn)
last_hidden = bert_outputs.last_hidden_state  # (batch_size, seq_len, hidden_size)

# Language modeling head: predict next token for each position
lm_logits = keras.layers.Dense(vocab_size)(last_hidden)  # (batch_size, seq_len, vocab_size)

# Define model
model = keras.Model(inputs=[ids, attn], outputs=lm_logits)

# 5. Compile with loss and optimizer
def masked_sparse_categorical_crossentropy(y_true, y_pred):
    # y_true: (batch_size, seq_len), y_pred: (batch_size, seq_len, vocab_size)
    loss = keras.losses.sparse_categorical_crossentropy(
        y_true, y_pred, from_logits=True
    )
    return tf.reduce_mean(loss)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=5e-5),
    loss=masked_sparse_categorical_crossentropy
)

# 6. Prepare labels: shift input_ids by one for next-token prediction
labels = tf.concat([input_ids[:, 1:], tf.zeros((input_ids.shape[0], 1), dtype=tf.int32)], axis=-1)

# 7. Train (example, with dummy epochs)
model.fit(
    {'input_ids': input_ids, 'attention_mask': attention_mask},
    labels,
    epochs=3,
    batch_size=2
)

# 8. Generate: given a prompt, predict next token
def predict_next_token(prompt, max_len=20):
    tokens = tokenizer(prompt, return_tensors='tf')
    for _ in range(max_len):
        out = model(tokens)[0]  # (seq_len, vocab_size)
        next_id = tf.argmax(out[-1]).numpy()
        tokens = {
            'input_ids': tf.concat([tokens['input_ids'], [[next_id]]], axis=-1),
            'attention_mask': tf.concat([tokens['attention_mask'], [[1]]], axis=-1)
        }
        if next_id == tokenizer.sep_token_id:
            break
    return tokenizer.decode(tokens['input_ids'][0])

# Example usage
print(predict_next_token("The capital of France is"))
