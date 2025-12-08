import os, argparse, numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def load_data(dataset):
    X = np.load(os.path.join('data', f"{dataset}-subset", 'X.npy'))
    y = np.load(os.path.join('data', f"{dataset}-subset", 'y.npy'))
    return X, y

def main(model_path, dataset='LSNM2024'):
    X, y = load_data(dataset)
    model = tf.keras.models.load_model(model_path, compile=False)
    preds = model.predict(X).ravel()
    preds_bin = (preds > 0.5).astype(int)
    print('Accuracy:', accuracy_score(y, preds_bin))
    print('F1:', f1_score(y, preds_bin))
    print('Precision:', precision_score(y, preds_bin))
    print('Recall:', recall_score(y, preds_bin))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='LSNM2024')
    args = parser.parse_args()
    main(args.model_path, args.dataset)
