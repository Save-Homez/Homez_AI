# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import psutil
import json

from dto import ReportRequestDto, ReportResponseDto, PointInfo, TimeRangeGroup
from dto import ListPointInfo, MLListRequestDto, MLListResponseDto
from model import load_model, predict_start_points
from model import load_mapping_data, encode_dest_point, decode_start_point
from model import load_station_mapping, load_travel_times, get_travel_time, get_station_from_dong
from matching import calculate_match_rate

app = Flask(__name__)

# 프로세스 객체 생성
process = psutil.Process()
# 모델 로드 전 메모리 사용량
mem_before = process.memory_info().rss
model = load_model()
# 모델 로드 후 메모리 사용량
mem_after = process.memory_info().rss

print(f"Memory used to load model: {(mem_after - mem_before) / 1024 ** 2:.2f} MB")
print("모델 로딩 완료")

# 데이터 매핑을 위한 데이터프레임 로드
dong_to_code_df, code_to_label_df = load_mapping_data()
station_mapping_df = load_station_mapping()
travel_times_df = load_travel_times()
print(travel_times_df.index.tolist())  # 인덱스 목록 출력
print(travel_times_df.columns.tolist())  # 컬럼 목록 출력


def logistic_scale(prob, scale=1):
    return 1 / (1 + np.exp(-prob * scale))

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# 사용자로부터 선호하는 요소와 기본 정보를 입력받아 사용자의 거주지를 추천하는 api입니다.
@app.post('/api/report/ml')
def predict_point():
    data = request.get_json()
    try:
        request_dto = ReportRequestDto(**data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # 입력 데이터에서 모델 입력 데이터 생성
    # Dow,Arrival_time,Start_pt,Dest_pt,Sex,Age,Move_time,Move_num,Wday
    dow = 0 # 수정 필요
    move_num = 0 # 수정 필요

    if request_dto.timeRange=="WITHIN_30_MINUTES":
        move_time = 30.0
    elif request_dto.timeRange=="WITHIN_60_MINUTES":
        move_time = 60.0
    elif request_dto.timeRange=="WITHIN_90_MINUTES":
        move_time = 90.0
    elif request_dto.timeRange=="WITHIN_120_MINUTES":
        move_time = 120.0
    else:
        move_time = 150.0

    # 도착지 라벨 인코딩에 맞게 매핑
    # 1. 동 이름 -> 코드
    # 2. 코드 -> 라벨 인코딩
    dest_point_encoded = encode_dest_point(request_dto.destPoint, dong_to_code_df, code_to_label_df)

    model_input = [
        dow,
        request_dto.arrivalTime,
        dest_point_encoded,
        request_dto.sex,
        request_dto.age,
        move_time,
        move_num,
        request_dto.workDay,
    ]

    # 각 클래스의 확률 예측
    probabilities = predict_start_points(model, np.array([model_input]))
    class_labels = model.classes_  # 모델이 학습한 클래스 라벨

    # 예측된 클래스 라벨을 동 이름으로 변환
    decoded_points = [decode_start_point(label, code_to_label_df, dong_to_code_df) for label in class_labels]

    # None 값을 제외하고 함수를 호출
    predicted_stations = [get_station_from_dong(dong, station_mapping_df) for dong in decoded_points if
                          dong is not None]

    # 결과 분류 및 처리
    point_list = []
    time_ranges = [30, 60, 90, 120, 150]
    previous_time_range = 0  # 이전 time range의 최대값 저장

    for idx, time_range in enumerate(time_ranges):
        relevant_points = []
        for dong, station, prob in zip(decoded_points, predicted_stations, probabilities):
            if station is not None and request_dto.destPoint is not None:
                print(station)
                # dest_station = get_station_from_dong(request_dto.destPoint, station_mapping_df)
                dest_station = request_dto.station
                if dest_station is not None:
                    print(dest_station)
                    try:
                        travel_time = get_travel_time(station, dest_station, travel_times_df)
                        # 이전 time range를 넘어서고 현재 time range 이하인지 확인
                        if previous_time_range < travel_time <= time_range:
                            relevant_points.append((dong, station, prob))
                    except ValueError:
                        continue  # 해당 역 이름에 대한 이동 시간 데이터가 없으면 무시
        # 다음 loop를 위해 현재 time range 저장
        previous_time_range = time_range

        # top_points = sorted(relevant_points, key=lambda x: x[2], reverse=True)[:3]
        top_points = sorted(relevant_points, key=lambda x: x[2], reverse=True)
        print(top_points)

        # None을 포함하지 않는 요소만 필터링하여 정렬
        filtered_points = [point for point in top_points if point[0] is not None]
        print(filtered_points)

        # 사용자 선호도에 따른 매칭율 계산 수행
        user_preferences = request_dto.factors
        # user_age = request_dto.age
        # user_sex = request_dto.sex

        # 최대 확률 값으로 나누어 스케일링
        max_prob = max(filtered_points, key=lambda x: x[2])[2] if filtered_points else 1
        normalized_probs = [prob / max_prob for _, _, prob in filtered_points]
        # 각 동에 대해 매칭율 계산 후 결과 튜플에 추가
        enhanced_points = []
        for (dong, station, prob), norm_prob in zip(filtered_points, normalized_probs):
            preferences_prob = calculate_match_rate(dong, user_preferences)
            weighted_prob = (norm_prob + preferences_prob) / 2
            enhanced_points.append((dong, station, prob, preferences_prob, weighted_prob))

        # # 각 동에 대해 매칭율 계산 후 결과 튜플에 추가
        # enhanced_points = []
        # for dong, station, prob in filtered_points:
        #     preferences_prob = calculate_match_rate(dong, user_preferences)
        #     # 제곱근 스케일링
        #     scaled_prob = np.sqrt(prob)
        #     weighted_prob = (scaled_prob + preferences_prob) / 2
        #     enhanced_points.append((dong, station, prob, preferences_prob, weighted_prob))

        # 필터링된 요소에 대해 정렬 수행
        # sorted_points = sorted(filtered_points, key=lambda x: x[2], reverse=True)[:10] # prob 기준
        sorted_points = sorted(enhanced_points, key=lambda x: x[4], reverse=True)[:10] # 선호 기준
        point_info_list = [
            PointInfo(name=dong, matchRate=f"{weighted_prob * 100:.2f}", rank=i + 1)
            for i, (dong, station, prob, preferences_prob, weighted_prob) in enumerate(sorted_points)
        ]
        point_list.append(TimeRangeGroup(timeRange=f"WITHIN_{time_range}_MINUTES", pointInfo=point_info_list))

    response_dto = ReportResponseDto(pointList=point_list)
    return jsonify(response_dto.dict())

@app.post('/api/report/ml-list')
def predict_point_list():
    data = request.get_json()
    try:
        request_dto = MLListRequestDto(**data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    # 입력 데이터에서 모델 입력 데이터 생성
    # Dow,Arrival_time,Start_pt,Dest_pt,Sex,Age,Move_time,Move_num,Wday
    dow = 0 # 수정 필요
    move_num = 0 # 수정 필요

    if request_dto.timeRange=="WITHIN_30_MINUTES":
        move_time = 30.0
    elif request_dto.timeRange=="WITHIN_60_MINUTES":
        move_time = 60.0
    elif request_dto.timeRange=="WITHIN_90_MINUTES":
        move_time = 90.0
    elif request_dto.timeRange=="WITHIN_120_MINUTES":
        move_time = 120.0
    else:
        move_time = 150.0

    # 도착지 라벨 인코딩에 맞게 매핑
    # 1. 동 이름 -> 코드
    # 2. 코드 -> 라벨 인코딩
    dest_point_encoded = encode_dest_point(request_dto.destPoint, dong_to_code_df, code_to_label_df)

    model_input = [
        dow,
        request_dto.arrivalTime,
        dest_point_encoded,
        request_dto.sex,
        request_dto.age,
        move_time,
        move_num,
        request_dto.workDay,
    ]

    # 각 클래스의 확률 예측
    probabilities = predict_start_points(model, np.array([model_input]))
    class_labels = model.classes_  # 모델이 학습한 클래스 라벨

    # 예측된 클래스 라벨을 동 이름으로 변환, None 값 제외
    decoded_points = [decode_start_point(label, code_to_label_df, dong_to_code_df) for label in class_labels]
    decoded_points = [point for point in decoded_points if point is not None]

    # 사용자 선호도에 따른 매칭율 계산, None 값 제외
    user_preferences = request_dto.factors

    # logistic_probabilities = [logistic_scale(prob) for prob in probabilities]
    #
    # point_list = []
    # for dong, log_prob in zip(decoded_points, logistic_probabilities):
    #     preferences_prob = calculate_match_rate(dong, user_preferences)
    #     weighted_prob = (log_prob + preferences_prob) / 2
    #     logistic_weighted_prob = logistic_scale(weighted_prob)
    #     point_list.append((dong, f"{logistic_weighted_prob * 100:.2f}%"))

    # softmax_probabilities = softmax(probabilities)
    #
    # point_list = []
    # for dong, softmax_prob in zip(decoded_points, softmax_probabilities):
    #     preferences_prob = calculate_match_rate(dong, user_preferences)
    #     weighted_prob = (softmax_prob + preferences_prob) / 2
    #     point_list.append((dong, f"{weighted_prob * 100:.2f}%"))

    # 확률 데이터를 시그모이드 함수로 변환
    sigmoid_probabilities = [sigmoid(prob) for prob in probabilities]

    point_list = []
    for dong, sigmoid_prob in zip(decoded_points, sigmoid_probabilities):
        preferences_prob = calculate_match_rate(dong, user_preferences)
        weighted_prob = (sigmoid_prob + preferences_prob) / 2
        point_list.append((dong, f"{weighted_prob * 100:.2f}"))

    # 결과 정렬 및 최종 DTO 생성
    sorted_points = sorted(point_list, key=lambda x: x[1], reverse=True)
    point_info_list = [ListPointInfo(name=dong, matchRate=matchRate) for dong, matchRate in sorted_points]

    response_dto = MLListResponseDto(pointList=point_info_list)
    return jsonify(response_dto.dict())

if __name__ == '__main__':
    app.run(port=5000, debug=True)