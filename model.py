# model.py
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (
    Bidirectional, LSTM, MultiHeadAttention,
    LayerNormalization, Dropout, Dense
)

def build_model(seq_len, feat_count, units1, units2, heads, dropout=0.2):
    inp = Input(shape=(seq_len, feat_count))
    x = Bidirectional(LSTM(units1, return_sequences=True))(inp)
    x = Bidirectional(LSTM(units2, return_sequences=True))(x)
    attn = MultiHeadAttention(num_heads=heads, key_dim=units2)(x, x)
    x = LayerNormalization()(x + attn)
    x = x[:, -1, :]
    x = Dropout(dropout)(x)
    out = Dense(1)(x)
    model = Model(inp, out)
    model.compile('adam', 'mse', metrics=['mae'])
    return model