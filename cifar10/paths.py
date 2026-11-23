"""Path resolution for the revision copy.

Input data (.pkl files, pretrained encoders, victim model) stays in its original
location and is not copied here. New outputs go to outputs/ in this folder, so the
original data is never overwritten.
"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Original experiment folder: .pkl data, pretrained encoders, victim model
DATA_DIR = '/home/didemgenc/Desktop/2024_SSL_MEA/Scenario4_MEA_SSL_KD/last_period/4k_budget_problem_Domain'
# Torchvision dataset root
DATASET_ROOT = '/home/didemgenc/Desktop/2024_SSL_MEA/data'

CKPT_DIR = os.path.join(HERE, 'checkpoints')
OUT_DIR = os.path.join(HERE, 'outputs')


def data_path(name):
    """Return the file from outputs/ if it was regenerated there, otherwise the original."""
    local = os.path.join(OUT_DIR, name)
    return local if os.path.exists(local) else os.path.join(DATA_DIR, name)


def out_path(name):
    """Path for newly generated data files (written under outputs/)."""
    path = os.path.join(OUT_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def ckpt_path(name):
    """Path for model checkpoints (read from / written to checkpoints/)."""
    path = os.path.join(CKPT_DIR, name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path
