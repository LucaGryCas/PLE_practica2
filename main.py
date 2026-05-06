import argparse
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, LSTM, Bidirectional, TimeDistributed
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nervaluate import Evaluator

# ── Parámetros por defecto ────────────────────────────────────────────────────
WINDOW_SIZE = 2
EMBED_DIM = 20
HIDDEN_DIM = 64
LSTM_DIM = 64
BATCH_SIZE = 32
EPOCHS = 5
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_TAG = "<PAD_TAG>"

# ── Lectura de datos ──────────────────────────────────────────────────────────
def read_sequence_labeling_data(path):
    sentences, labels = [], []
    current_tokens, current_tags = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_tags)
                    current_tokens, current_tags = [], []
                continue

            if "\t" in line:
                parts = line.split("\t")
            else:
                parts = line.split()

            if len(parts) < 2:
                raise ValueError(
                    f"Formato inesperado en {path}, línea {line_number}: {line!r}"
                )

            token = parts[0]
            tag = parts[-1]
            current_tokens.append(token)
            current_tags.append(tag)

    if current_tokens:
        sentences.append(current_tokens)
        labels.append(current_tags)

    return sentences, labels



# ── Vocabularios ──────────────────────────────────────────────────────────────
def build_word_vocab(sentences, min_freq=1):
    counter = Counter(token for sent in sentences for token in sent)
    word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}

    for word, freq in counter.items():
        if freq >= min_freq and word not in word2idx:
            word2idx[word] = len(word2idx)

    return word2idx

def build_tag_vocab(labels):
    unique_tags = sorted({tag for sent_tags in labels for tag in sent_tags})
    tag2idx = {tag: i for i, tag in enumerate(unique_tags)}
    idx2tag = {i: tag for tag, i in tag2idx.items()}
    return tag2idx, idx2tag

# ── Snippets ──────────────────────────────────────────────────────────────────
def create_snippets(sentences, labels, word2idx, tag2idx, window_size=2):
    X, y = [], []
    pad_id = word2idx[PAD_TOKEN]
    unk_id = word2idx[UNK_TOKEN]

    for sent, tag_seq in zip(sentences, labels):
        sent_ids = [word2idx.get(t, unk_id) for t in sent]
        padded = [pad_id] * window_size + sent_ids + [pad_id] * window_size

        for i in range(window_size, len(sent_ids) + window_size):
            X.append(padded[i - window_size : i + window_size + 1])
            y.append(tag2idx[tag_seq[i - window_size]])

    return np.array(X, dtype=np.int32), np.array(y, dtype=np.int32)

# ── Preparación de datos para LSTM/BiLSTM con oraciones completas ───────────── (LSTM trabaja con secuencias completas no snippets)

def encode_sentences(sentences, word2idx):
    " Convierte palabras en IDs "
    unk_id = word2idx[UNK_TOKEN]
    return [[word2idx.get(token, unk_id) for token in sent] for sent in sentences]

def encode_labels(labels, tag2idx):
    " Etiquetas en IDs "
    return [[tag2idx[tag] for tag in tag_seq] for tag_seq in labels]

def create_lstm_data(sentences, labels, word2idx, tag2idx, max_len=None):
    " Convierte las oraciones en IDs y les añade padding para poder usarlas en la LSTM "
    X = encode_sentences(sentences, word2idx)
    y = encode_labels(labels, tag2idx)

    if max_len is None:
        max_len = max(len(sent) for sent in X)

    X_pad = pad_sequences(
        X,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=word2idx[PAD_TOKEN],
    )
    y_pad = pad_sequences(
        y,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=tag2idx[PAD_TAG],
    )

    return X_pad.astype(np.int32), y_pad.astype(np.int32), max_len

# ── Pesos para entrenamiento y evaluación ─────────────────────────────────────

def compute_tag_weights(train_labels, tag2idx, task):
    " Calcula el peso de cada etiqueta"
    weights = np.ones(len(tag2idx), dtype=np.float32)

    if task != "ner":
        return weights

    counter = Counter(tag for sent in train_labels for tag in sent)
    total = sum(counter.values())

    for tag, idx in tag2idx.items():
        if tag == PAD_TAG: # Ignoramos el padding
            weights[idx] = 0.0
        elif counter[tag] > 0:
            weights[idx] = total / (len(counter) * counter[tag])

    return weights


def make_sample_weights(y, tag_weights, pad_tag_id=0):
    " Asigna a cada token el peso correspondiente de su etiqueta "
    weights = tag_weights[y]
    weights = np.where(y == pad_tag_id, 0.0, weights)
    return weights.astype(np.float32)


# ── Modelos ───────────────────────────────────────────────────────────────────
def build_ff_model(vocab_size, window_size, n_tags):
    inputs = Input(shape=(2 * window_size + 1,), name="tokens")
    x = Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, name="embedding")(inputs)
    x = Flatten(name="flatten")(x)
    x = Dense(HIDDEN_DIM, activation="relu", name="hidden_dense")(x)
    outputs = Dense(n_tags, activation="softmax", name="tag_classifier")(x)

    model = Model(inputs=inputs, outputs=outputs, name="ff_snippet_tagger")
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(vocab_size, max_len, n_tags, bidirectional=False):
    inputs = Input(shape=(max_len,), name="tokens")
    x = Embedding(
        input_dim=vocab_size,
        output_dim=EMBED_DIM,
        mask_zero=True,
        name="embedding",
    )(inputs)

    lstm_layer = LSTM(LSTM_DIM, return_sequences=True, name="lstm")
    if bidirectional:
        x = Bidirectional(lstm_layer, name="bilstm")(x)
    else:
        x = lstm_layer(x)

    outputs = TimeDistributed(
        Dense(n_tags, activation="softmax"),
        name="tag_classifier_each_token",
    )(x)

    model_name = "bilstm_sequence_tagger" if bidirectional else "lstm_sequence_tagger"
    model = Model(inputs=inputs, outputs=outputs, name=model_name)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ── Predicción y evaluación ──────────────────────────────────────────────────

def predict_ff_sentence_tags(model, sentences, word2idx, idx2tag, window_size):
    all_pred_tags = []
    pad_id = word2idx[PAD_TOKEN]
    unk_id = word2idx[UNK_TOKEN]

    for sent in sentences:
        sent_ids = [word2idx.get(token, unk_id) for token in sent]
        padded = [pad_id] * window_size + sent_ids + [pad_id] * window_size
        snippets = []

        for i in range(window_size, len(sent_ids) + window_size):
            snippets.append(padded[i - window_size : i + window_size + 1])

        probs = model.predict(np.array(snippets, dtype=np.int32), verbose=0)
        pred_ids = np.argmax(probs, axis=-1)
        all_pred_tags.append([idx2tag[pred_id] for pred_id in pred_ids])

    return all_pred_tags


def predict_lstm_sentence_tags(model, sentences, word2idx, idx2tag, max_len):
    X = encode_sentences(sentences, word2idx)
    X_pad = pad_sequences(
        X,
        maxlen=max_len,
        padding="post",
        truncating="post",
        value=word2idx[PAD_TOKEN],
    )
    probs = model.predict(X_pad.astype(np.int32), verbose=0)
    pred_ids = np.argmax(probs, axis=-1)

    all_pred_tags = []
    for sent, pred_seq in zip(sentences, pred_ids):
        all_pred_tags.append([idx2tag[pred_id] for pred_id in pred_seq[: len(sent)]])

    return all_pred_tags


def token_accuracy(y_true_tags, y_pred_tags):
    correct = 0
    total = 0

    for true_seq, pred_seq in zip(y_true_tags, y_pred_tags):
        for true_tag, pred_tag in zip(true_seq, pred_seq):
            correct += int(true_tag == pred_tag)
            total += 1

    return correct / total if total else 0.0


def evaluate_ner_nervaluate(y_true, y_pred):
    entity_tags = sorted(
        {
            tag.split("-", 1)[1]
            for sent in y_true
            for tag in sent
            if tag != "O" and "-" in tag
        }
    )

    if not entity_tags:
        print("\nNo se han encontrado entidades BIO para evaluar con nervaluate.")
        return

    evaluator = Evaluator(y_true, y_pred, tags=entity_tags, loader="list")
    results, results_by_tag = evaluator.evaluate()

    print("\n" + "=" * 60)
    print("Evaluación NER con nervaluate")
    print("=" * 60)

    for schema in ["strict", "exact", "partial", "ent_type"]:
        res = results[schema]
        print(f"\n[{schema}]")
        print(f"Precision: {res['precision']:.4f}")
        print(f"Recall:    {res['recall']:.4f}")
        print(f"F1:        {res['f1']:.4f}")

    print("\n" + "=" * 60)
    print("Resultados por tipo de entidad")
    print("=" * 60)

    for entity, entity_results in results_by_tag.items():
        print(f"\nEntidad: {entity}")
        for schema in ["strict", "exact", "partial", "ent_type"]:
            res = entity_results[schema]
            print(
                f"  {schema:8s} -> "
                f"P: {res['precision']:.4f} | "
                f"R: {res['recall']:.4f} | "
                f"F1: {res['f1']:.4f}"
            )

# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Sequence labeling con Keras")
    parser.add_argument("--model", required=True, choices=["ff", "lstm", "bilstm"])
    parser.add_argument("--task", required=True, choices=["ner", "pos"])
    parser.add_argument("--train", required=True, metavar="PATH")
    parser.add_argument("--dev", required=True, metavar="PATH")
    parser.add_argument("--test", required=True, metavar="PATH")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch", type=int, default=BATCH_SIZE)
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'=' * 60}")
    print(f"Tarea  : {args.task.upper()}")
    print(f"Modelo : {args.model.upper()}")
    print(f"Ventana: n={args.window}")
    print(f"{'=' * 60}\n")

    train_sents, train_labels = read_sequence_labeling_data(args.train)
    dev_sents, dev_labels = read_sequence_labeling_data(args.dev)
    test_sents, test_labels = read_sequence_labeling_data(args.test)

    word2idx = build_word_vocab(train_sents)
    tag2idx, idx2tag = build_tag_vocab(train_labels)
    tag_weights = compute_tag_weights(train_labels, tag2idx, args.task)

    print(f"Oraciones train/dev/test: {len(train_sents)} / {len(dev_sents)} / {len(test_sents)}")
    print(f"Vocabulario            : {len(word2idx)} tokens")
    print(f"Etiquetas              : {len(tag2idx)} tags -> {list(tag2idx.keys())}\n")

    if args.model == "ff":
        X_train, y_train = create_snippets(train_sents, train_labels, word2idx, tag2idx, args.window)
        X_dev, y_dev = create_snippets(dev_sents, dev_labels, word2idx, tag2idx, args.window)
        X_test, y_test = create_snippets(test_sents, test_labels, word2idx, tag2idx, args.window)

        train_weights = make_sample_weights(y_train, tag_weights)
        dev_weights = make_sample_weights(y_dev, tag_weights)

        model = build_ff_model(len(word2idx), args.window, len(tag2idx))
        model.summary()
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            validation_data=(X_dev, y_dev, dev_weights),
            epochs=args.epochs,
            batch_size=args.batch,
            verbose=1,
        )

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        y_pred_tags = predict_ff_sentence_tags(model, test_sents, word2idx, idx2tag, args.window)

    else:
        X_train, y_train, max_len = create_lstm_data(train_sents, train_labels, word2idx, tag2idx)
        X_dev, y_dev, _ = create_lstm_data(dev_sents, dev_labels, word2idx, tag2idx, max_len=max_len)
        X_test, y_test, _ = create_lstm_data(test_sents, test_labels, word2idx, tag2idx, max_len=max_len)

        train_weights = make_sample_weights(y_train, tag_weights)
        dev_weights = make_sample_weights(y_dev, tag_weights)
        test_weights = make_sample_weights(y_test, np.ones(len(tag2idx), dtype=np.float32))

        model = build_lstm_model(
            len(word2idx),
            max_len,
            len(tag2idx),
            bidirectional=(args.model == "bilstm"),
        )
        model.summary()
        model.fit(
            X_train,
            y_train,
            sample_weight=train_weights,
            validation_data=(X_dev, y_dev, dev_weights),
            epochs=args.epochs,
            batch_size=args.batch,
            verbose=1,
        )

        loss, acc = model.evaluate(X_test, y_test, sample_weight=test_weights, verbose=0)
        y_pred_tags = predict_lstm_sentence_tags(model, test_sents, word2idx, idx2tag, max_len)

    acc_real = token_accuracy(test_labels, y_pred_tags)
    print(f"\nTest Keras -> loss: {loss:.4f} | acc: {acc:.4f}")
    print(f"Test real  -> accuracy token a token sin padding: {acc_real:.4f}")

    if args.task == "ner":
        evaluate_ner_nervaluate(test_labels, y_pred_tags)


if __name__ == "__main__":
    main()

