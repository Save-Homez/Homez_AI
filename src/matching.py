import pandas as pd

# 정규화된 CSV 파일 로드
dong_features = pd.read_csv('../data/normalized_preferences.csv', index_col='동 이름')


def calculate_match_rate(dong, user_preferences):
    # dong의 특성 데이터 가져오기
    try:
        dong_data = dong_features.loc[dong].to_dict()
    except KeyError:
        # dong 키가 없는 경우, 매칭 불가로 0을 반환
        return 0

    total_score = sum(dong_data.get(pref, 0) for pref in user_preferences)
    max_score = len(user_preferences)  # 각 선호 요소에 가중치 1을 부여
    match_rate = (total_score / max_score) if max_score != 0 else 0
    return match_rate
