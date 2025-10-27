import pandas as pd
import json
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
import re

print("=" * 80)
print("GEMINI 모델 설득력 기반 전략 분석")
print("=" * 80)
print()

# === 1. 데이터 로드 및 전처리 ===
print("1. 데이터 로드 및 전처리")
print("-" * 40)

# 기만 전략 분석 데이터 로드
df = pd.read_csv('deception_strategy_analysis.csv')
gemini_df = df[df['model'] == 'gemini-2.5-pro'].copy()

# 새로운 Gemini 매핑 적용
with open('gemini_strategy_mapping.json', 'r', encoding='utf-8') as f:
    gemini_mapping = json.load(f)

gemini_df['final_strategy'] = gemini_df['mapped_strategy'].map(gemini_mapping)
gemini_df.dropna(subset=['final_strategy'], inplace=True)

print(f"✓ Gemini 전략 데이터: {len(gemini_df):,}개 레이블링")

# 설득력 데이터 로드
def parse_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    json_strings = re.findall(r'{.*?}', content, re.DOTALL)
    data = []
    for s in json_strings:
        try:
            data.append(json.loads(s))
        except json.JSONDecodeError:
            continue
    return data

persuasion_data = parse_json_lines('matched_sample.json')
persuasion_df = pd.DataFrame(persuasion_data)

# 설득력 데이터 전처리
persuasion_metric_key = 'persuasiveness_metric' if 'persuasiveness_metric' in persuasion_df.columns else 'persuasion_metric'
persuasion_df.rename(columns={'argument': 'original_argument', persuasion_metric_key: 'persuasion_metric'}, inplace=True)
persuasion_df = persuasion_df[['original_argument', 'persuasion_metric']].drop_duplicates()

print(f"✓ 설득력 데이터: {len(persuasion_df)}개 논증")

# 데이터 병합
if 'argument' in gemini_df.columns:
    gemini_df.rename(columns={'argument': 'original_argument'}, inplace=True)

gemini_df = pd.merge(gemini_df, persuasion_df, on='original_argument', how='left')
gemini_df['argument_id'] = pd.factorize(gemini_df['original_argument'])[0]

# 중복 제거 (같은 버전이 같은 논증에서 같은 전략을 여러 번 언급)
gemini_df_unique = gemini_df.drop_duplicates(subset=['argument_id', 'version', 'final_strategy'])
print(f"✓ 중복 제거 후: {len(gemini_df_unique):,}개 고유 레이블링")

# 설득력 점수 분포 확인
persuasion_dist = gemini_df_unique['persuasion_metric'].value_counts().sort_index()
print(f"✓ 설득력 점수 분포:")
for score, count in persuasion_dist.items():
    if pd.notna(score):
        print(f"  점수 {score}: {count}개")

print()

# === 2. 설득력 수준별 분류 ===
print("2. 설득력 수준별 분류")
print("-" * 40)

# 설득력 점수를 3단계로 분류
def classify_persuasion(score):
    if pd.isna(score):
        return 'Unknown'
    elif score <= 1:
        return 'Low'
    elif score <= 2:
        return 'Medium'
    else:
        return 'High'

gemini_df_unique['persuasion_level'] = gemini_df_unique['persuasion_metric'].apply(classify_persuasion)

level_dist = gemini_df_unique['persuasion_level'].value_counts()
print("설득력 수준별 분포:")
for level, count in level_dist.items():
    percentage = (count / len(gemini_df_unique)) * 100
    print(f"  {level}: {count}개 ({percentage:.1f}%)")

print()

# === 3. 설득력 수준별 전략 사용 빈도 분석 ===
print("3. 설득력 수준별 전략 사용 빈도 분석")
print("-" * 40)

# 설득력 수준별 전략 분포
persuasion_strategy_crosstab = pd.crosstab(gemini_df_unique['persuasion_level'], 
                                          gemini_df_unique['final_strategy'], 
                                          normalize='index') * 100

print("설득력 수준별 전략 분포 (%):")
print(persuasion_strategy_crosstab.round(1))
print()

# 고설득력 vs 저설득력 전략 비교
high_persuasion = gemini_df_unique[gemini_df_unique['persuasion_level'] == 'High']['final_strategy'].value_counts(normalize=True) * 100
low_persuasion = gemini_df_unique[gemini_df_unique['persuasion_level'] == 'Low']['final_strategy'].value_counts(normalize=True) * 100

print("🔥 고설득력 논증에서 가장 많이 사용되는 전략:")
for strategy, percentage in high_persuasion.head(5).items():
    print(f"  {strategy}: {percentage:.1f}%")

print()
print("❄️ 저설득력 논증에서 가장 많이 사용되는 전략:")
for strategy, percentage in low_persuasion.head(5).items():
    print(f"  {strategy}: {percentage:.1f}%")

print()

# === 4. 논증별 전략 조합 분석 ===
print("4. 논증별 전략 조합 분석")
print("-" * 40)

# 각 논증별로 사용된 전략들의 집합 생성
argument_strategies = gemini_df_unique.groupby(['argument_id', 'persuasion_level'])['final_strategy'].agg(set).reset_index()
argument_strategies['strategy_count'] = argument_strategies['final_strategy'].apply(len)
argument_strategies['strategy_list'] = argument_strategies['final_strategy'].apply(lambda x: sorted(list(x)))

print("설득력 수준별 평균 전략 수:")
avg_strategies = argument_strategies.groupby('persuasion_level')['strategy_count'].agg(['mean', 'std']).round(2)
for level in avg_strategies.index:
    mean_val = avg_strategies.loc[level, 'mean']
    std_val = avg_strategies.loc[level, 'std']
    print(f"  {level}: {mean_val:.2f} ± {std_val:.2f}개")

print()

# === 5. 효과적인 전략 조합 분석 ===
print("5. 효과적인 전략 조합 분석")
print("-" * 40)

# 고설득력 논증에서 자주 나타나는 전략 조합
high_persuasion_args = argument_strategies[argument_strategies['persuasion_level'] == 'High']

if len(high_persuasion_args) > 0:
    print("🔥 고설득력 논증에서 자주 나타나는 전략 조합:")
    
    # 2개 전략 조합 분석
    strategy_pairs = []
    for strategies in high_persuasion_args['final_strategy']:
        if len(strategies) >= 2:
            for pair in combinations(sorted(strategies), 2):
                strategy_pairs.append(pair)
    
    if strategy_pairs:
        pair_counter = Counter(strategy_pairs)
        print("\n가장 자주 나타나는 2개 전략 조합:")
        for pair, count in pair_counter.most_common(10):
            percentage = (count / len(high_persuasion_args)) * 100
            print(f"  {pair[0]} + {pair[1]}: {count}회 ({percentage:.1f}%)")
    
    # 3개 전략 조합 분석
    strategy_triplets = []
    for strategies in high_persuasion_args['final_strategy']:
        if len(strategies) >= 3:
            for triplet in combinations(sorted(strategies), 3):
                strategy_triplets.append(triplet)
    
    if strategy_triplets:
        triplet_counter = Counter(strategy_triplets)
        print("\n가장 자주 나타나는 3개 전략 조합:")
        for triplet, count in triplet_counter.most_common(5):
            percentage = (count / len(high_persuasion_args)) * 100
            print(f"  {triplet[0]} + {triplet[1]} + {triplet[2]}: {count}회 ({percentage:.1f}%)")

print()

# === 6. 전략별 설득력 효과 분석 ===
print("6. 전략별 설득력 효과 분석")
print("-" * 40)

# 각 전략이 포함된 논증들의 평균 설득력 점수
strategy_effectiveness = []

for strategy in gemini_df_unique['final_strategy'].unique():
    strategy_data = gemini_df_unique[gemini_df_unique['final_strategy'] == strategy]
    
    # 해당 전략이 포함된 논증들의 설득력 점수
    strategy_args = strategy_data.groupby('argument_id')['persuasion_metric'].first().dropna()
    
    if len(strategy_args) > 0:
        avg_persuasion = strategy_args.mean()
        strategy_effectiveness.append({
            'strategy': strategy,
            'avg_persuasion': avg_persuasion,
            'argument_count': len(strategy_args)
        })

effectiveness_df = pd.DataFrame(strategy_effectiveness).sort_values('avg_persuasion', ascending=False)

print("전략별 평균 설득력 점수 (높은 순):")
for _, row in effectiveness_df.iterrows():
    print(f"  {row['strategy']}: {row['avg_persuasion']:.2f} (논증 {row['argument_count']}개)")

print()

# === 7. 설득력 점수별 상세 분석 ===
print("7. 설득력 점수별 상세 분석")
print("-" * 40)

# 설득력 점수별 전략 분포
for score in sorted(gemini_df_unique['persuasion_metric'].dropna().unique()):
    score_data = gemini_df_unique[gemini_df_unique['persuasion_metric'] == score]
    strategy_dist = score_data['final_strategy'].value_counts(normalize=True) * 100
    
    print(f"설득력 점수 {score} (총 {len(score_data)}개):")
    for strategy, percentage in strategy_dist.head(3).items():
        print(f"  {strategy}: {percentage:.1f}%")
    print()

# === 8. 결과 저장 ===
print("8. 결과 저장")
print("-" * 40)

# 설득력별 전략 분포 저장
persuasion_strategy_crosstab.to_csv('gemini_persuasion_strategy_distribution.csv', encoding='utf-8-sig')
print("✓ 설득력별 전략 분포: 'gemini_persuasion_strategy_distribution.csv'")

# 논증별 전략 조합 저장
argument_strategies_export = argument_strategies.copy()
argument_strategies_export['strategy_list'] = argument_strategies_export['strategy_list'].apply(lambda x: ', '.join(x))
argument_strategies_export.drop('final_strategy', axis=1, inplace=True)
argument_strategies_export.to_csv('gemini_argument_strategy_combinations.csv', index=False, encoding='utf-8-sig')
print("✓ 논증별 전략 조합: 'gemini_argument_strategy_combinations.csv'")

# 전략별 효과성 저장
effectiveness_df.to_csv('gemini_strategy_effectiveness.csv', index=False, encoding='utf-8-sig')
print("✓ 전략별 효과성: 'gemini_strategy_effectiveness.csv'")

print()
print("=" * 80)
print("분석 완료! 설득력 높은 논증에서 자주 사용되는 전략과 조합을 확인했습니다.")
print("=" * 80) 