"""Train the DNN channel estimator (Fig. 3: 80 -> 256 -> 128 -> 14, ReLU, Adam, MSE).

Usage:  python train.py --ep 14        (one network per pilot power Ep/Es in dB)
        python train.py --ep 23
"""
import argparse
import time
import joblib
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import otfs

# Learning-rate schedule after the first stage: (epochs, learning rate)
FINETUNE = [(20, 3e-4), (20, 1e-4), (10, 3e-5)]


def make_dataset(n, rng, ep):
    """Random channels at the training SNR -> (DNN inputs, labels)."""
    kmax = otfs.kmax_from_speed(otfs.MAX_SPEED_KMH)
    X, Y = [], []
    for _ in range(n // 10_000):                                        # build in chunks of 10k
        ch = otfs.random_channels(10_000, rng, kmax, p_single=0.2)
        S = otfs.simulate(10_000, rng, ch, otfs.TRAIN_SNR_DB, ep)
        X.append(otfs.dnn_input(S["Y"], S["xp"]))
        Y.append(otfs.encode_label(ch, otfs.phase_ref(S["Y"], S["xp"])))
    return np.concatenate(X), np.concatenate(Y)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ep", type=int, required=True, help="pilot power Ep/Es [dB]")
    p.add_argument("--samples", type=int, default=400_000, help="first-stage training samples")
    p.add_argument("--epochs", type=int, default=60, help="first-stage epochs")
    args = p.parse_args()

    # Step 1: generate the training set
    t = time.time()
    X, Y = make_dataset(args.samples, np.random.default_rng(args.ep), args.ep)
    print(f"dataset {X.shape} -> {Y.shape} in {time.time() - t:.0f}s")

    # Step 2: build the network (inputs and targets are standardised)
    net = MLPRegressor(hidden_layer_sizes=otfs.HIDDEN, activation="relu", solver="adam",
                       learning_rate_init=1e-3, batch_size=256, max_iter=args.epochs,
                       early_stopping=False, tol=1e-6, n_iter_no_change=20, random_state=0)
    model = TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(), net),
                                       transformer=StandardScaler())

    # Step 3: first training stage (lr = 1e-3)
    t = time.time()
    model.fit(X, Y)
    net = model.regressor_[-1]                                          # the fitted network (a copy)
    print(f"stage 1: {net.n_iter_} epochs in {time.time() - t:.0f}s, loss {net.loss_:.5f}")

    # Step 4: fine-tune with decaying learning rate on fresh data
    in_scaler, out_scaler = model.regressor_[0], model.transformer_
    for epochs, lr in FINETUNE:
        rng = np.random.default_rng(1000 + args.ep + len(net.loss_curve_))
        X, Y = make_dataset(200_000, rng, args.ep)
        net.set_params(learning_rate_init=lr, warm_start=True, max_iter=epochs)
        net._no_improvement_count = 0                                   # don't stop early
        t = time.time()
        net.fit(in_scaler.transform(X), out_scaler.transform(Y))        # continue training
        print(f"fine-tune lr={lr:g}: {epochs} epochs in {time.time() - t:.0f}s, loss {net.loss_:.5f}")

    # Step 5: save
    joblib.dump(dict(model=model, loss=net.loss_curve_), f"models/dnn_ep{args.ep}.joblib")
    print(f"saved models/dnn_ep{args.ep}.joblib")


if __name__ == "__main__":
    main()
