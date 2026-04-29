# =============================================================================
# Sarcasm Detection - LSTM Advanced Model
# Input: sarcasm_train.csv & sarcasm_test.csv
# =============================================================================

# =============================================================================
# SECTION 1 - INSTALL & IMPORT LIBRARIES
# =============================================================================
# Run this in terminal first:
# pip install tensorflow pandas numpy scikit-learn

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import (accuracy_score, f1_score,
                             confusion_matrix, classification_report)

# =============================================================================
# SECTION 2 - LOAD TRAIN & TEST DATA
# =============================================================================
print("=" * 60)
print("SECTION 2: LOADING DATA")
print("=" * 60)

train = pd.read_csv("sarcasm_train.csv")
test  = pd.read_csv("sarcasm_test.csv")

print("Train size:", len(train))
print("Test size:", len(test))

# =============================================================================
# SECTION 3 - PREPARE COMBINED TEXT & LABELS
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 3: PREPARING TEXT & LABELS")
print("=" * 60)

# Create combined text if not already in the file
if "combined_text" not in train.columns:
    train["combined_text"] = train["comment"] + " " + train["parent_comment"]
    test["combined_text"]  = test["comment"]  + " " + test["parent_comment"]
    print("Combined text created!")
else:
    print("Combined text already exists!")

X_train = train["combined_text"]
X_test  = test["combined_text"]

y_train = train["label"].values
y_test  = test["label"].values

print("\nClass balance in train:")
print(pd.Series(y_train).value_counts(normalize=True).round(3))
print("\nClass balance in test:")
print(pd.Series(y_test).value_counts(normalize=True).round(3))

# =============================================================================
# SECTION 4 - TOKENIZATION
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 4: TOKENIZATION")
print("=" * 60)

vocab_size = 10000   # top 10000 most common words

# Fit tokenizer on TRAIN only (prevent data leakage)
tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Convert text to sequences
train_seq = tokenizer.texts_to_sequences(X_train)
test_seq  = tokenizer.texts_to_sequences(X_test)

print("Vocabulary size:", len(tokenizer.word_index))
print("Example sequence:", train_seq[0][:10], "...")

# =============================================================================
# SECTION 5 - PADDING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 5: PADDING")
print("=" * 60)

maxlen = 100   # max words per comment

train_pad = pad_sequences(train_seq, maxlen=maxlen,
                          padding="post", truncating="post")
test_pad  = pad_sequences(test_seq,  maxlen=maxlen,
                          padding="post", truncating="post")

print("Train padded shape:", train_pad.shape)
print("Test padded shape:", test_pad.shape)

# =============================================================================
# SECTION 6 - BUILD LSTM MODEL
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 6: BUILDING LSTM MODEL")
print("=" * 60)

model = Sequential([
    Embedding(input_dim    = vocab_size,  # vocabulary size
              output_dim   = 128,          # embedding dimensions
              input_length = maxlen),      # sequence length

    LSTM(units=64, return_sequences=False),# LSTM layer
    Dropout(0.5),                          # prevent overfitting

    Dense(units=1, activation="sigmoid")  # binary output (0 or 1)
])

model.compile(
    optimizer = "adam",
    loss      = "binary_crossentropy",    # as per proposal
    metrics   = ["accuracy"]
)

model.summary()

# =============================================================================
# SECTION 7 - TRAIN LSTM MODEL WITH EARLY STOPPING
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 7: TRAINING LSTM MODEL")
print("=" * 60)

# Early stopping — stops when val_loss stops improving
# patience=3 because model takes ~3 epochs to start learning
early_stop = EarlyStopping(
    monitor              = "val_loss",
    patience             = 3,             # stop after 3 bad epochs
    restore_best_weights = True,          # revert to best epoch automatically
    verbose              = 1
)

# Model checkpoint — saves best model to file automatically
checkpoint = ModelCheckpoint(
    filepath       = "best_lstm_model.keras",
    monitor        = "val_loss",          # save when val_loss improves
    save_best_only = True,
    verbose        = 1
)

print("Early stopping : ON (patience=3)")
print("Model checkpoint: ON (saves best_lstm_model.keras)")
print("Starting training...\n")

history = model.fit(
    train_pad, y_train,
    epochs           = 10,
    batch_size       = 32,               # increased from 32 (faster training)
    validation_split = 0.2,              # 20% of train for validation
    callbacks        = [early_stop, checkpoint],
    verbose          = 1
)

best_epoch = np.argmin(history.history["val_loss"]) + 1
print(f"\nBest epoch      : {best_epoch}")
print(f"Best val_loss   : {min(history.history['val_loss']):.4f}")
print(f"Best val_accuracy: {max(history.history['val_accuracy']):.4f}")

# =============================================================================
# SECTION 8 - EVALUATE LSTM MODEL
# =============================================================================
print("\n" + "=" * 60)
print("SECTION 8: EVALUATING LSTM MODEL")
print("=" * 60)

# Predict
predictions = (model.predict(test_pad) > 0.5).astype(int).flatten()

# Accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Test Accuracy : {accuracy:.4f}")

# F1 Score
f1 = f1_score(y_test, predictions)
print(f"F1 Score      : {f1:.4f}")

# Full classification report
print("\nClassification Report:")
print(classification_report(y_test, predictions,
      target_names=["Not Sarcastic (0)", "Sarcastic (1)"]))

# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("(Rows = Actual, Columns = Predicted)")
