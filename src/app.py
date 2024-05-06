# Import libraries
import numpy as np
from flask import Flask, request, jsonify
import pickle
import json

from dto import ReportRequestDto, ReportResponseDto, PointInfo, TimeRangeGroup
from model import load_model, predict_start_points
from model import load_mapping_data, encode_dest_point, decode_start_point
from model import load_station_mapping, load_travel_times, get_travel_time, get_station_from_dong

app = Flask(__name__)

# 모델 초기화
model = load_model()
print("모델 로딩 완료")

# 데이터 매핑을 위한 데이터프레임 로드
dong_to_code_df, code_to_label_df = load_mapping_data()
station_mapping_df = load_station_mapping()
travel_times_df = load_travel_times()

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
    for time_range in [30, 60, 90, 120, 150]:
        relevant_points = []
        for dong, station, prob in zip(decoded_points, predicted_stations, probabilities):
            if station is not None and request_dto.destPoint is not None:
                print(station)
                dest_station = get_station_from_dong(request_dto.destPoint, station_mapping_df)
                if dest_station is not None:
                    print(dest_station)
                    try:
                        travel_time = get_travel_time(station, dest_station, travel_times_df)
                        if travel_time <= time_range:
                            relevant_points.append((dong, station, prob))
                    except ValueError:
                        continue  # 해당 역 이름에 대한 이동 시간 데이터가 없으면 무시
        top_points = sorted(relevant_points, key=lambda x: x[2], reverse=True)[:3]
        point_info_list = [
            PointInfo(name=dong, matchRate=f"{prob * 100:.2f}%", rank=i + 1)
            for i, (dong, station, prob) in enumerate(top_points)
        ]
        point_list.append(TimeRangeGroup(timeRange=f"Within {time_range} minutes", pointInfo=point_info_list))

    response_dto = ReportResponseDto(pointList=point_list)
    return jsonify(response_dto.dict())


if __name__ == '__main__':
    app.run(port=5000, debug=True)