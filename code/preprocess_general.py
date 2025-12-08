import numpy as np
import pandas as pd
import os
import os, argparse

def generate_representative_subset(outdir, samples=1000, input_dim=50, seq_len=20, seed=42):
    os.makedirs(outdir, exist_ok=True)
    rng = np.random.RandomState(seed)
    # create synthetic sequences with some structure:
    # benign sequences: Gaussian noise centered at 0
    # malicious sequences: Gaussian noise centered at +1 on a fraction of features
    X = rng.randn(samples, seq_len, input_dim).astype('float32')
    # inject simple pattern for half of samples to simulate attacks
    half = samples // 2
    X[half:] += 1.0  # shift mean
    y = np.zeros((samples,), dtype='int')
    y[half:] = 1
    np.save(os.path.join(outdir, 'X.npy'), X)
    np.save(os.path.join(outdir, 'y.npy'), y)
    print(f"Wrote subset to {outdir} (samples={samples}, seq_len={seq_len}, input_dim={input_dim})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='LSNM2024', choices=['LSNM2024','DoHBrw2020','CICIoT2023'])
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--input_dim', type=int, default=50)
    parser.add_argument('--seq_len', type=int, default=20)
    args = parser.parse_args()
    out = os.path.join('data', f"{args.dataset}-subset")
    generate_representative_subset(out, samples=args.samples, input_dim=args.input_dim, seq_len=args.seq_len)
