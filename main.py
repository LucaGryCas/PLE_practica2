import argparse
import numpy as np
import tensorflow as tf
from collections import Counter
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, LSTM, Bidirectional
from tensorflow.keras.models import Model
from nervaluate import Evaluator

# ── Parámetros por defecto ────────────────────────────────────────────────────
WINDOW_SIZE = 2
EMBED_DIM   = 20
HIDDEN_DIM  = 64
BATCH_SIZE  = 32
EPOCHS      = 5
PAD_TOKEN   = "<PAD>"
UNK_TOKEN   = "<UNK>"

# ── Lectura de datos ──────────────────────────────────────────────────────────
def read_sequence_labeling_data(path):
    sentences, labels = [], []
    current_tokens, current_tags = [], []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                if current_tokens:
                    sentences.append(current_tokens)
                    labels.append(current_tags)
                    current_tokens, current_tags = [], []
                continue

            parts = line.split("\t")
            if len(parts) != 2:
                raise ValueError(f"Formato inesperado en {path}: {line!r}")

            token, tag = parts
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
        if freq >= min_freq:
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

# ── Predicción por oración para evaluación NER ────────────────────────────────
def predict_sentence_tags(model, sentences, word2idx, idx2tag, window_size=2):
    all_pred_tags = []
    pad_id = word2idx[PAD_TOKEN]
    unk_id = word2idx[UNK_TOKEN]

    for sent in sentences:
        sent_ids = [word2idx.get(t, unk_id) for t in sent]
        padded = [pad_id] * window_size + sent_ids + [pad_id] * window_size

        X_sent = []
        for i in range(window_size, len(sent_ids) + window_size):
            snippet = padded[i - window_size : i + window_size + 1]
            X_sent.append(snippet)

        X_sent = np.array(X_sent, dtype=np.int32)
        pred_probs = model.predict(X_sent, verbose=0)
        pred_ids = np.argmax(pred_probs, axis=1)
        pred_tags = [idx2tag[i] for i in pred_ids]

        all_pred_tags.append(pred_tags)

    return all_pred_tags

# ── Modelo ────────────────────────────────────────────────────────────────────
def build_model(vocab_size, window_size, n_tags,
                embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                rnn=None, lstm_dim=64):
    """rnn: None | 'lstm' | 'bilstm'"""
    inputs = Input(shape=(2 * window_size + 1,))
    x = Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    if rnn == "lstm":
        x = LSTM(lstm_dim, return_sequences=False)(x)
    elif rnn == "bilstm":
        x = Bidirectional(LSTM(lstm_dim, return_sequences=False))(x)
    else:
        x = Flatten()(x)

    x = Dense(hidden_dim, activation="relu")(x)
    outputs = Dense(n_tags, activation="softmax")(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ── Evaluación NER con nervaluate ─────────────────────────────────────────────
def evaluate_ner_nervaluate(y_true, y_pred):
    # Extrae los tipos de entidad sin prefijos BIO
    entity_tags = sorted({
        tag.split("-", 1)[1]
        for sent in y_true
        for tag in sent
        if tag != "O" and "-" in tag
    })

    evaluator = Evaluator(y_true, y_pred, tags=entity_tags, loader="list")
    results, results_by_tag = evaluator.evaluate()

    print("\n" + "=" * 60)
    print("Evaluación NER con nervaluate")
    print("=" * 60)

    for schema in ["strict", "exact", "partial", "ent_type"]:
        if schema in results:
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
            if schema in entity_results:
                res = entity_results[schema]
                print(
                    f"  {schema:8s} -> "
                    f"P: {res['precision']:.4f} | "
                    f"R: {res['recall']:.4f} | "
                    f"F1: {res['f1']:.4f}"
                )

# ── Main ──────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Sequence labeling con redes neuronales")
    parser.add_argument("--model", required=True, choices=["ff", "lstm", "bilstm"],
                        help="Arquitectura del modelo")
    parser.add_argument("--task", required=True, choices=["ner", "pos"],
                        help="Tarea de etiquetado")
    parser.add_argument("--train", required=True, metavar="PATH", help="Fichero de entrenamiento")
    parser.add_argument("--dev", required=True, metavar="PATH", help="Fichero de desarrollo")
    parser.add_argument("--test", required=True, metavar="PATH", help="Fichero de test")
    parser.add_argument("--window", type=int, default=WINDOW_SIZE, help="Tamaño de ventana (default: 2)")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Épocas (default: 5)")
    parser.add_argument("--batch", type=int, default=BATCH_SIZE, help="Batch size (default: 32)")
    return parser.parse_args()

def main():
    args = parse_args()

    print(f"\n{'='*50}")
    print(f"  Tarea  : {args.task.upper()}")
    print(f"  Modelo : {args.model.upper()}")
    print(f"  Ventana: n={args.window}")
    print(f"{'='*50}\n")

    # Lectura
    train_sents, train_labels = read_sequence_labeling_data(args.train)
    dev_sents, dev_labels = read_sequence_labeling_data(args.dev)
    test_sents, test_labels = read_sequence_labeling_data(args.test)

    # Vocabularios (solo train)
    word2idx = build_word_vocab(train_sents)
    tag2idx, idx2tag = build_tag_vocab(train_labels)

    print(f"Vocabulario : {len(word2idx):>6} tokens")
    print(f"Etiquetas   : {len(tag2idx):>6} tags → {list(tag2idx.keys())}\n")

    # Snippets
    rnn_flag = None if args.model == "ff" else args.model
    ws = args.window

    X_train, y_train = create_snippets(train_sents, train_labels, word2idx, tag2idx, ws)
    X_dev, y_dev = create_snippets(dev_sents, dev_labels, word2idx, tag2idx, ws)
    X_test, y_test = create_snippets(test_sents, test_labels, word2idx, tag2idx, ws)

    # Modelo
    model = build_model(
        vocab_size=len(word2idx),
        window_size=ws,
        n_tags=len(tag2idx),
        rnn=rnn_flag
    )
    model.summary()

    # Entrenamiento
    model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        epochs=args.epochs,
        batch_size=args.batch,
        verbose=1
    )

    # Evaluación básica token a token
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nTest → loss: {loss:.4f} | acc: {acc:.4f}")

    # Evaluación NER real por oración
    if args.task == "ner":
        y_pred_ner = predict_sentence_tags(model, test_sents, word2idx, idx2tag, ws)
        evaluate_ner_nervaluate(test_labels, y_pred_ner)

if __name__ == "__main__":
    main()
