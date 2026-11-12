# DNN-Based Fractional Doppler Channel Estimation for OTFS
Reproduced by **Hania Tariq** 

A Python reproduction of

> L. Guo, P. Gu, J. Zou, G. Liu and F. Shu, "DNN-Based Fractional Doppler Channel Estimation for OTFS Modulation," *IEEE Trans. Veh. Technol.*, vol. 72, no. 11, pp. 15062–15067, Nov. 2023, doi: 10.1109/TVT.2023.3280901.

In high-mobility links, such as LEO satellites and UAVs at up to 1000 km/h, the Doppler shift falls between OTFS grid points. Its energy then spreads across the delay-Doppler grid, which makes the channel hard to estimate. The paper trains a small DNN that reads the received pilot and estimates only a few channel parameters per path: gain, delay and fractional Doppler. The full channel matrix is rebuilt from those parameters and passed to an MRC detector.

This repository implements the whole link from scratch in NumPy:
- the zero-padded OTFS system;
- a two-path channel with fractional Doppler;
- the DNN estimator;
- the conventional delay-time pilot estimator as the baseline;
- the delay-time MRC detector of Thaj & Viterbo (*IEEE TVT*, 2020).

It reproduces Figs. 4–7 of the paper.

## System parameters

| Parameter | Value | How it is obtained |
|---|---|---|
| Grid size M × N | 64 × 8 | paper |
| Subcarrier spacing, carrier | 15 kHz, 3 GHz | paper |
| Modulation | 4-QAM | paper |
| Max delay tap l_max | 4 | 4 µs ÷ delay resolution 1/(MΔf) = 1.04 µs |
| Max Doppler k_max | 1.48 bins | 2778 Hz at 1000 km/h ÷ Doppler resolution Δf/N = 1875 Hz |
| Guard rows L_ZP | 9 | 2 l_max + 1 |
| DNN | 80 → 256 → 128 → 14 | inputs 2(l_max+1)N, outputs 2(1+3P) with P = 2 |
| Training | ReLU, Adam, MSE loss, SNR 10 dB | paper |

## Results

| NMSE (Fig. 4) | BER, two paths (Fig. 5) |
|---|---|
| ![](figures/fig4_nmse.png) | ![](figures/fig5_ber.png) |
| **BER vs speed (Fig. 6)** | **BER, one path (Fig. 7)** |
| ![](figures/fig6_ber_speed.png) | ![](figures/fig7_ber_one_path.png) |

### Comparison with the paper

Paper values are read approximately from its figures. Each cell gives paper / this repo.

**NMSE, two paths (Fig. 4)**

| Curve | 8 dB | 12 dB | 16 dB |
|---|---|---|---|
| Conventional, Ep/Es = 11 dB | 0.5 / 0.57 | – / 0.23 | 0.08 / 0.096 |
| Conventional, Ep/Es = 20 dB | 0.065 / 0.078 | 0.031 / 0.035 | 0.016 / 0.019 |
| Conventional, Ep/Es = 30 dB | 0.012 / 0.014 | 0.0085 / 0.0099 | 0.0065 / 0.0085 |
| DNN, Ep/Es = 14 dB | 0.015 / 0.042 | 0.0072 / 0.014 | 0.0049 / 0.0095 |
| DNN, Ep/Es = 23 dB | 0.005 / 0.0074 | 0.0037 / 0.0027 | 0.003 / 0.0018 |
| DNN + LS gains, Ep/Es = 14 dB *(not in paper)* | – / 0.028 | – / 0.010 | – / 0.0054 |

**BER, one path (Fig. 7)**

| Curve | 8 dB | 10 dB | 12 dB |
|---|---|---|---|
| Perfect CSI | 6.0e-3 / 6.0e-3 | 8.0e-4 / 7.7e-4 | 4.0e-5 / 3.6e-5 |
| Conventional, Ep/Es = 40 dB | 7.0e-3 / 6.7e-3 | 1.2e-3 / 1.0e-3 | 1.4e-4 / 8.5e-5 |
| DNN, Ep/Es = 14 dB | 9.5e-3 / 8.8e-3 | 1.4e-3 / 1.2e-3 | 1.4e-4 / 1.1e-4 |
| DNN, Ep/Es = 23 dB | 7.0e-3 / 6.4e-3 | 1.1e-3 / 8.3e-4 | 8.5e-5 / 4.2e-5 |

**BER, two paths (Fig. 5)**

| Curve | 8 dB | 12 dB | 16 dB |
|---|---|---|---|
| Perfect CSI | 6.5e-3 / 1.3e-2 | 5.0e-4 / 2.5e-4 | 2.5e-5 / 1.0e-6 |
| Conventional, Ep/Es = 30 dB | 8.0e-3 / 1.6e-2 | 7.0e-4 / 5.5e-4 | 6.5e-5 / 1.3e-5 |
| DNN, Ep/Es = 14 dB | 1.0e-2 / 2.7e-2 | 8.5e-4 / 8.9e-4 | 6.5e-5 / 9.1e-5 |
| DNN, Ep/Es = 23 dB | 7.5e-3 / 1.5e-2 | 6.5e-4 / 3.3e-4 | 3.2e-5 / 3.2e-6 |

**Reproduced**
- **Fig. 7 (one path):** perfect CSI matches to two digits. The conventional and DNN curves are within about 0.8–1.1× of the paper, except some 12 dB points where ours is lower.
- **Conventional estimator (Fig. 4):** within 1.1–1.3× of the paper at every point.
- **DNN at Ep/Es = 23 dB:** matches or beats the paper's NMSE and is close to perfect CSI in BER.
- **Pilot power:** the DNN reaches a given accuracy at much lower pilot power than the conventional method, and raising the conventional pilot from 30 to 40 dB gives almost nothing.
- **Speed:** one network, trained up to 1000 km/h, works at both 500 and 1000 km/h (Fig. 6).

**Not reproduced**
- **DNN at Ep/Es = 14 dB:** about 2–3× higher NMSE than the paper. The error comes mainly from the gain estimates. With the true delays and Dopplers, least-squares gains reach NMSE 0.013 / 0.005 / 0.002 at 8 / 12 / 16 dB. The paper's DNN is close to that limit; our network is not. Re-estimating the gains by least squares from the DNN's delays and Dopplers (*DNN + LS*, our addition, not in the paper) reaches the paper's value at 16 dB.
- **BER of the DNN at Ep/Es = 14 dB (Fig. 5):** levels off near 1e-4 above 14 dB. Almost all errors come from a few frames where two equally strong paths and a small Doppler error make the MRC detector diverge.
- **Two-path perfect-CSI curve (Fig. 5):** ours is about 2× worse at 8 dB and much better at 16 dB. Every two-path BER curve inherits this shift. The paper does not give enough channel or detector detail to match this curve; the one-path curve, which depends only on the channel, does match.
- **Not implemented:** the OMP, data-aided pilot and two-pilot baselines.

### Assumptions (not specified in the paper)
- **Channel (calibrated to the paper's figures):**
  - Two Rayleigh paths, with gains normalised to total power 1 in every frame. The one-path case (path-2 gain = 0) then gives |h| = 1, which matches the paper's Fig. 7 perfect-CSI curve (4-QAM in AWGN).
  - Path 1 at delay 0, path 2 at a random delay of 1–4 taps.
  - Doppler of each path uniform in [−k_max, k_max].
  - 20 % of training frames have a single path.
- **Conventional estimator:** the pilot gives the gain of all 5 delay taps at N time instants, with no tap threshold. A natural cubic spline then interpolates across the frame. This reproduces the paper's conventional NMSE curves.
- **DNN output:** 14 values: real and imaginary parts of {h1, h2, k1, k2, l1, l2, P}. Delays and P are rounded.
- **Pre-processing (added):** each received pilot row is phase-aligned before entering the DNN, and the rotation is undone afterwards. Inputs and targets are standardised.
- **Training schedule:** 400k samples for 60 epochs at learning rate 1e-3, then 20/20/10 epochs at 3e-4/1e-4/3e-5 on fresh data; batch 256.
- **MRC detector:** 15 iterations, the first 5 without hard decisions.
- **Framework:** scikit-learn `MLPRegressor`, with the same architecture as the paper.

## How to run

```bash
pip install -r requirements.txt

python -m pytest tests          # sanity checks (~1 s)
python train.py --ep 14         # train (optional, models/ already has trained networks, ~25 min each)
python train.py --ep 23
python evaluate.py --fast       # simulations for Figs. 4-7 (drop --fast for the full run)
python plot.py                  # figures -> figures/
```

No GPU is needed.

## Repository structure

```
otfs/
  config.py        system parameters
  channel.py       two-path fractional-Doppler channel, channel matrix g[l, q]
  system.py        OTFS transmitter, channel, receiver
  estimators.py    DNN input/output encoding, LS gain refinement, conventional estimator, NMSE
  detector.py      delay-time MRC detector, bit errors
train.py           dataset generation and DNN training
evaluate.py        NMSE / BER simulations -> results/results.json
plot.py            figures
tests/             simulator sanity checks
models/            trained networks (Ep/Es = 14 dB and 23 dB)
results/           simulation results
figures/           reproduced figures
```

## References
1. L. Guo et al., "DNN-Based Fractional Doppler Channel Estimation for OTFS Modulation," *IEEE Trans. Veh. Technol.*, vol. 72, no. 11, pp. 15062–15067, 2023.
2. T. Thaj and E. Viterbo, "Low Complexity Iterative Rake Decision Feedback Equalizer for Zero-Padded OTFS Systems," *IEEE Trans. Veh. Technol.*, vol. 69, no. 12, pp. 15606–15622, 2020.
3. 3GPP TR 38.811, "Study on New Radio (NR) to support non-terrestrial networks."


[def]: https://github.com/HANIA-Tariq