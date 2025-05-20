# Torch imports
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import (
    efficientnet_b0,
    EfficientNet_B0_Weights,
    efficientnet_b2,
    EfficientNet_B2_Weights,
)
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter


# Other imports
from typing import Dict, List, Tuple
from timeit import default_timer as timer
import random
from PIL import Image
from tqdm.auto import tqdm
import requests
import zipfile
from pathlib import Path
import os  # listing data content of a directory
import random
from PIL import Image  # visualizing images of dataset
import matplotlib.pyplot as plt
import sys  # enabling python to import from another dir

sys.path.append("/Users/lorenzofabbri/Downloads/GitHub/Going_Modular/going_modular")
from helper_functions import plot_loss_curves, set_seeds

# Modular imports
import data_setup
import engine

# import model_builder
# import train


# A bit of magic to make pretrained model import work
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
