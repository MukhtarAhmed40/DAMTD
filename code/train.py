import os, yaml, argparse
import numpy as np
import tensorflow as tf
from code.models import build_convbilstm_mha
from code.domain_adaptation import GradientReversalLayer, maximum_mean_discrepancy

def load_data(dataset):
    X = np.load(os.path.join('data', f"{dataset}-subset", 'X.npy'))
    y = np.load(os.path.join('data', f"{dataset}-subset", 'y.npy'))
    return X, y

def build_domain_discriminator(feature_dim):
    inputs = tf.keras.Input(shape=(feature_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(32, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inputs, out, name='domain_discriminator')

def main(cfg):
    dataset = cfg.get('dataset','LSNM2024')
    batch_size = cfg.get('batch_size', 64)
    epochs = cfg.get('epochs', 5)
    lr = cfg.get('learning_rate', 1e-3)
    seq_len = cfg.get('seq_len', 20)
    input_dim = cfg.get('input_dim', 50)
    model_save = cfg.get('model_save_path', f'pretrained_models/damtd_{dataset}.h5')
    use_mmd = cfg.get('mmd_lambda', 0.0) > 0.0
    use_adv = cfg.get('adv_lambda', 0.0) > 0.0

    X, y = load_data(dataset)
    # simple train/test split
    n = X.shape[0]
    idx = np.random.permutation(n)
    train_idx = idx[:int(0.8*n)]
    val_idx = idx[int(0.8*n):]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    model = build_convbilstm_mha((seq_len, input_dim), cnn_filters=cfg.get('cnn_filters',32), lstm_units=cfg.get('lstm_units',64), heads=cfg.get('attention_heads',4))
    # compile and train
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
    print(model.summary())

    # fit
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)

    # save model
    os.makedirs(os.path.dirname(model_save), exist_ok=True)
    model.save(model_save)
    print('Saved model to', model_save)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/damtd_LSNM2024.yaml')
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    main(cfg)
