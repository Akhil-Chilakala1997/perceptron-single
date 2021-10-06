
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
import os

def prepare_data(df):
  
  x= df.drop("y", axis=1)

  y = df["y"]

  return x, y

def save_model(model, filename):
  model_dir = "models"
  os.makedirs(model_dir, exist_ok=True) # ONLY CREATE IF MODEL_DIR DOESN"T EXISTS
  filePath = os.path.join(model_dir, filename) # model/filename
  joblib.dump(model, filePath)
  