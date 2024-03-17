import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import string
import re
from scipy.stats import zscore
from scipy.stats import spearmanr
from sklearn.preprocessing import  OneHotEncoder

directory = 'data/FinalblurbMerged_kickstarter_data.xlsx'
            #'Test_Judge.xlsx'
df = pd.read_excel(directory)