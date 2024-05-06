# Import libraries
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
import pickle

def load_model():
    with open('../model/xgb_model_0505.pkl', 'rb') as file:
        model = pickle.load(file)
    return model
# 동 이름을 코드로, 코드를 라벨로 매핑하는 데이터를 로드
def load_mapping_data():
    dong_to_code_df = pd.read_csv('../data/merge_dong_code.csv')
    code_to_label_df = pd.read_csv('../data/label_mapping_0505.csv')
    return dong_to_code_df, code_to_label_df

# 동 이름을 입력으로 받아 라벨 인코딩된 값을 반환
def encode_dest_point(dong_name, dong_to_code_df, code_to_label_df):
    # 동 이름에서 코드 찾기
    code = dong_to_code_df.loc[dong_to_code_df['동 이름'] == dong_name, '읍면동'].values[0]
    # 코드에서 라벨 인코딩 찾기
    encoded_label = code_to_label_df.loc[code_to_label_df['OriginalLabel'] == code, 'EncodedLabel'].values[0]
    return encoded_label

# 동 코드를 동 이름으로 변환
def decode_start_point(encoded_label, code_to_label_df, dong_to_code_df):
    # 라벨 인코딩에서 동 코드 찾기
    original_code = code_to_label_df.loc[code_to_label_df['EncodedLabel'] == encoded_label, 'OriginalLabel'].values[0]
    # 동 코드에서 동 이름 찾기
    dong_name = dong_to_code_df.loc[dong_to_code_df['읍면동'] == original_code, '동 이름'].values[0]
    return dong_name
def predict_start_points(model, input_data):
    # 모델을 사용하여 startPoint와 해당 확률을 예측
    probabilities = model.predict_proba(input_data)
    return probabilities

# 동 이름에서 역 이름으로 매핑하는 데이터 로드
def load_station_mapping():
    station_mapping_df = pd.read_csv('../data/station_dong.csv')
    return station_mapping_df

# 역 간 이동 시간 데이터 로드
def load_travel_times():
    travel_times_df = pd.read_csv('../data/shortest_path_costs.csv')
    return travel_times_df

# 동 이름을 입력받아 해당하는 역 이름 반환
def get_station_from_dong(dong_name, station_mapping_df):
    station_name = station_mapping_df.loc[station_mapping_df['동네'] == dong_name, 'station'].values[0]
    return station_name

# 두 역 이름을 입력받아 이동 시간 반환
def get_travel_time(start_station, dest_station, travel_times_df):
    travel_time = travel_times_df.loc[(travel_times_df['from'] == start_station) & (travel_times_df['to'] == dest_station), 'time'].values[0]
    return travel_time