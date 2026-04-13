import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

import src.data_preparation as dp

FILE_PATH = 'data/data.csv'

def main():
    data = dp.process_dataset(FILE_PATH, save_processed=True)
    