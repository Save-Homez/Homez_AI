# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle

# Load the model
model = pickle.load(open('../model/xgb_model.pkl', 'rb'))

