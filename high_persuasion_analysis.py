import pandas as pd
import json
from collections import Counter
from itertools import combinations

def parse_json_lines(file_path):
    """JSON Lines 파일을 파싱합니다."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
        return []
    return data

# === 1. 데이터 로드 ===
print("1. 데이터 로드")
print("-" * 40)

# 전략 매핑 파일 로드
with open('gemini_strategy_mapping.json', 'r', encoding='utf-8') as f:
    strategy_mapping = json.load(f)

# 최종 8개 전략 매핑
final_taxonomy_mapping = {
    "감정 조작": "Appealing to Emotion",
    "권위 오용": "Misusing Authority", 
    "증거 왜곡·선택적 제시": "Misusing Evidence",
    "논점 전환·왜곡": "Attacking or Evading the Point",
    "추론 오류": "Using Flawed Reasoning",
    "프레이밍": "Framing for Perception Manipulation",
    "사회적 통념에 호소": "Appealing to Social Norms",
    "불확실성 관련 전략": "Using Uncertainty"
}

# 분석 데이터 로드
df = pd.read_csv('final_deception_analysis.csv')
# 이미 mapped_strategy와 final_strategy가 있으므로 그대로 사용
df.dropna(subset=['final_strategy'], inplace=True)

# Gemini 데이터만 필터링
gemini_df = df[df['model'] == 'gemini-2.5-pro'].copy()
print(f"✓ Gemini 데이터: {len(gemini_df):,}개 레이블링")

# 설득력 데이터 로드
persuasion_data = parse_json_lines('matched_sample.json')
persuasion_df = pd.DataFrame(persuasion_data)
persuasion_metric_key = 'persuasiveness_metric' if 'persuasiveness_metric' in persuasion_df.columns else 'persuasion_metric'
persuasion_df.rename(columns={'argument': 'original_argument', persuasion_metric_key: 'persuasion_metric'}, inplace=True)
persuasion_df = persuasion_df[['original_argument', 'persuasion_metric']]
print(f"✓ 설득력 데이터: {len(persuasion_df)}개 논증")

# 데이터 병합
if 'argument' in gemini_df.columns:
    gemini_df.rename(columns={'argument': 'original_argument'}, inplace=True)

gemini_df = pd.merge(gemini_df, persuasion_df, on='original_argument', how='left')
gemini_df['argument_id'] = pd.factorize(gemini_df['original_argument'])[0]

# 중복 제거
gemini_df_unique = gemini_df.drop_duplicates(subset=['argument_id', 'version', 'final_strategy'])
print(f"✓ 중복 제거 후: {len(gemini_df_unique):,}개 고유 레이블링")

print()

# === 2. 설득력 점수 분포 확인 ===
print("2. 설득력 점수 분포 확인")
print("-" * 40)

# 논증별 설득력 점수 확인
argument_persuasion = gemini_df_unique.groupby('argument_id')['persuasion_metric'].first().dropna()
persuasion_dist = argument_persuasion.value_counts().sort_index()

print("논증별 설득력 점수 분포:")
for score, count in persuasion_dist.items():
    percentage = (count / len(argument_persuasion)) * 100
    print(f"  점수 {score}: {count}개 논증 ({percentage:.1f}%)")

# 3점 이상인 논증 확인
high_persuasion_args = argument_persuasion[argument_persuasion >= 3]
print(f"\n✓ 설득력 점수 3점 이상인 논증: {len(high_persuasion_args)}개")
print(f"✓ 해당 논증들의 설득력 점수 평균: {high_persuasion_args.mean():.2f}")
print(f"✓ 설득력 점수 범위: {high_persuasion_args.min():.1f} ~ {high_persuasion_args.max():.1f}")

print()

# === 3. 고설득력 논증 분석 ===
print("3. 고설득력 논증 분석 (설득력 점수 3점 이상)")
print("-" * 40)

if len(high_persuasion_args) == 0:
    print("❌ 설득력 점수가 3점 이상인 논증이 없습니다.")
    print("   가장 높은 설득력 점수:", argument_persuasion.max())
else:
    # 고설득력 논증의 argument_id 리스트
    high_persuasion_arg_ids = high_persuasion_args.index.tolist()
    
    # 해당 논증들의 전략 레이블링만 필터링
    high_persuasion_labelings = gemini_df_unique[gemini_df_unique['argument_id'].isin(high_persuasion_arg_ids)]
    
    print(f"✓ 고설득력 논증에서 나온 전략 레이블링: {len(high_persuasion_labelings):,}개")
    print(f"✓ 평균 설득력 점수 (검증): {high_persuasion_labelings['persuasion_metric'].mean():.2f}")
    
    # 전략 분포 분석
    strategy_dist = high_persuasion_labelings['final_strategy'].value_counts()
    strategy_percentage = (strategy_dist / strategy_dist.sum()) * 100
    
    print(f"\n고설득력 논증에서 사용된 전략 분포:")
    for strategy, count in strategy_dist.items():
        percentage = strategy_percentage[strategy]
        print(f"  {strategy}: {count}개 ({percentage:.1f}%)")
    
    # 논증별 전략 조합 분석
    argument_strategies = high_persuasion_labelings.groupby('argument_id').agg({
        'final_strategy': lambda x: set(x),
        'persuasion_metric': 'first',
        'original_argument': 'first'
    }).reset_index()
    
    argument_strategies['strategy_count'] = argument_strategies['final_strategy'].apply(len)
    argument_strategies['strategy_list'] = argument_strategies['final_strategy'].apply(lambda x: sorted(list(x)))
    
    print(f"\n논증별 전략 사용 현황:")
    print(f"✓ 평균 전략 수: {argument_strategies['strategy_count'].mean():.2f}개")
    print(f"✓ 전략 수 범위: {argument_strategies['strategy_count'].min()}개 ~ {argument_strategies['strategy_count'].max()}개")
    
    # 전략 수별 분포
    strategy_count_dist = argument_strategies['strategy_count'].value_counts().sort_index()
    print(f"\n전략 수별 논증 분포:")
    for count, freq in strategy_count_dist.items():
        percentage = (freq / len(argument_strategies)) * 100
        print(f"  {count}개 전략: {freq}개 논증 ({percentage:.1f}%)")
    
    # 가장 효과적인 전략 조합 찾기
    print(f"\n가장 자주 나타나는 전략 조합:")
    
    # 2개 전략 조합
    strategy_pairs = []
    for strategies in argument_strategies['final_strategy']:
        if len(strategies) >= 2:
            for pair in combinations(sorted(strategies), 2):
                strategy_pairs.append(pair)
    
    if strategy_pairs:
        pair_counter = Counter(strategy_pairs)
        print(f"\n2개 전략 조합 (상위 5개):")
        for pair, count in pair_counter.most_common(5):
            percentage = (count / len(argument_strategies)) * 100
            print(f"  {pair[0]} + {pair[1]}: {count}회 ({percentage:.1f}%)")
    
    # 3개 전략 조합
    strategy_triplets = []
    for strategies in argument_strategies['final_strategy']:
        if len(strategies) >= 3:
            for triplet in combinations(sorted(strategies), 3):
                strategy_triplets.append(triplet)
    
    if strategy_triplets:
        triplet_counter = Counter(strategy_triplets)
        print(f"\n3개 전략 조합 (상위 3개):")
        for triplet, count in triplet_counter.most_common(3):
            percentage = (count / len(argument_strategies)) * 100
            print(f"  {triplet[0]} + {triplet[1]} + {triplet[2]}: {count}회 ({percentage:.1f}%)")
    
    # 결과 저장
    print(f"\n4. 결과 저장")
    print("-" * 40)
    
    # 고설득력 논증 상세 정보 저장
    argument_strategies_export = argument_strategies.copy()
    argument_strategies_export['strategy_list_str'] = argument_strategies_export['strategy_list'].apply(lambda x: ', '.join(x))
    argument_strategies_export.drop(['final_strategy', 'strategy_list'], axis=1, inplace=True)
    argument_strategies_export.to_csv('high_persuasion_arguments_analysis.csv', index=False, encoding='utf-8-sig')
    print("✓ 고설득력 논증 분석: 'high_persuasion_arguments_analysis.csv'")
    
    # 전략 분포 저장
    strategy_dist_df = pd.DataFrame({
        'strategy': strategy_dist.index,
        'count': strategy_dist.values,
        'percentage': strategy_percentage.values
    })
    strategy_dist_df.to_csv('high_persuasion_strategy_distribution.csv', index=False, encoding='utf-8-sig')
    print("✓ 전략 분포: 'high_persuasion_strategy_distribution.csv'")

print()
print("=" * 80)
print("분석 완료!") 