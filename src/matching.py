import pandas as pd

# 정규화된 CSV 파일 로드
dong_features = pd.read_csv('../data/normalized_preferences.csv', index_col='동 이름')


def calculate_match_rate(dong, user_preferences):
    try:
        dong_data = dong_features.loc[dong].to_dict()
    except KeyError:
        return 0

    total_score = 0
    for pref in user_preferences:
        value = dong_data.get(pref, 0)
        if isinstance(value, (int, float)):  # 값 유효성 검사
            total_score += value
        else:
            continue  # 유효하지 않은 값은 무시

    max_score = len(user_preferences)
    match_rate = (total_score / max_score) if max_score != 0 else 0
    return match_rate

