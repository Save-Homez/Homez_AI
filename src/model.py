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
    # 정확히 일치하는 동 이름 검색
    match_exact = dong_to_code_df[dong_to_code_df['동 이름'] == dong_name]

    if not match_exact.empty:
        code = match_exact['읍면동'].values[0]
    else:
        # 정확한 일치 결과가 없으면 포함하는 문자열 검색
        match_contains = dong_to_code_df[dong_to_code_df['동 이름'].str.contains(dong_name)]
        if not match_contains.empty:
            code = match_contains['읍면동'].values[0]
        else:
            # 일치하는 동 이름이 전혀 없는 경우 오류 처리
            raise ValueError(f"No matching entry for dong name '{dong_name}' in the dataset.")

    # 코드에서 라벨 인코딩 찾기
    encoded_label = code_to_label_df.loc[code_to_label_df['OriginalLabel'] == code, 'EncodedLabel'].values[0]
    return encoded_label

# 동 코드를 동 이름으로 변환
def decode_start_point(encoded_label, code_to_label_df, dong_to_code_df):
    # 라벨 인코딩에서 동 코드 찾기 시도
    original_code = code_to_label_df.loc[code_to_label_df['EncodedLabel'] == encoded_label, 'OriginalLabel'].values
    if len(original_code) == 0:
        return None  # 또는 "Unknown code" 등의 메시지 반환

    original_code = original_code[0]  # 첫 번째 일치 항목 사용

    # 동 코드에서 동 이름 찾기
    match_exact = dong_to_code_df[dong_to_code_df['읍면동'] == original_code]
    if not match_exact.empty:
        dong_name = match_exact['동 이름'].values[0]
    else:
        # 정확한 일치 결과가 없으면 포함하는 문자열 검색
        match_contains = dong_to_code_df[dong_to_code_df['읍면동'].astype(str).str.contains(str(original_code))]
        if not match_contains.empty:
            dong_name = match_contains['동 이름'].values[0]
        else:
            return None  # 또는 "Unknown location" 등의 메시지 반환

    return dong_name

def predict_start_points(model, input_data):
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
    if not isinstance(dong_name, str) or dong_name is None:
        return None  # 동 이름이 유효한 문자열이 아니면 None을 반환

    try:
        # 동 이름에 기반하여 역 이름 찾기
        station_name = station_mapping_df.loc[station_mapping_df['동네'] == dong_name, 'station'].values[0]
        return station_name
    except IndexError:
        # 정확한 일치 결과가 없으면 포함하는 문자열 검색
        match_contains = station_mapping_df[station_mapping_df['동네'].str.contains(dong_name, na=False)]
        if not match_contains.empty:
            return match_contains['station'].values[0]
        # 일치하는 역 이름이 전혀 없는 경우
        return None


# 두 역 이름을 입력받아 이동 시간 반환
# def get_travel_time(start_station, dest_station, travel_times_df):
#     try:
#         # 두 역 이름을 사용하여 데이터프레임에서 이동 시간 조회
#         travel_time = travel_times_df.loc[start_station, dest_station]
#         return travel_time
#     except KeyError:
#         # 시작역이나 도착역 이름이 데이터프레임에 없는 경우 오류 처리
#         raise ValueError(f"Travel time data unavailable for stations: {start_station} or {dest_station}.")

def get_travel_time(start_station, dest_station, travel_times_df):
    # 역 이름의 부분 문자열로 해당 역을 포함하는 첫 번째 역 이름을 찾는 과정
    def find_station_containing(substring, df):
        filtered = df.columns[df.columns.str.contains(substring, case=False, regex=True, na=False)]
        return filtered[0] if not filtered.empty else None

    start_station_match = find_station_containing(start_station, travel_times_df)
    dest_station_match = find_station_containing(dest_station, travel_times_df)

    if not start_station_match or not dest_station_match:
        raise ValueError(f"Travel time data unavailable for stations: {start_station} or {dest_station}.")

    try:
        travel_time = travel_times_df.loc[start_station_match, dest_station_match]
        return travel_time
    except KeyError:
        raise ValueError(f"Travel time data unavailable for stations: {start_station_match} or {dest_station_match}.")

