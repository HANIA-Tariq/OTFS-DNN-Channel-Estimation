"""OTFS simulator with DNN-based fractional Doppler channel estimation."""
from .config import *
from .channel import kmax_from_speed, random_channels, channel_matrix
from .system import simulate
from .estimators import phase_ref, dnn_input, encode_label, decode_output, ls_gains, conventional_estimate, nmse
from .detector import mrc_detect, bit_errors
