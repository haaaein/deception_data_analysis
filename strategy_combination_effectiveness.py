import pandas as pd
import json
import numpy as np
from collections import Counter
from itertools import combinations
import re

print("=" * 80)
print("전략 조합별 설득력 효과 분석")
print("어떤 조합이 가장 설득력이 높은가?")
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

# 설득력 점수가 있는 데이터만 필터링
gemini_df_with_scores = gemini_df_unique.dropna(subset=['persuasion_metric'])
print(f"✓ 설득력 점수 있는 데이터: {len(gemini_df_with_scores):,}개")

print()

# === 2. 논증별 전략 조합 생성 ===
print("2. 논증별 전략 조합 생성")
print("-" * 40)

# 각 논증별로 사용된 전략들의 집합과 설득력 점수
argument_strategies = gemini_df_with_scores.groupby('argument_id').agg({
    'final_strategy': lambda x: set(x),
    'persuasion_metric': 'first',
    'original_argument': 'first'
}).reset_index()

argument_strategies['strategy_count'] = argument_strategies['final_strategy'].apply(len)
argument_strategies['strategy_list'] = argument_strategies['final_strategy'].apply(lambda x: tuple(sorted(x)))

print(f"✓ 분석 대상 논증 수: {len(argument_strategies)}개")
print(f"✓ 평균 전략 수: {argument_strategies['strategy_count'].mean():.2f}개")
print(f"✓ 평균 설득력 점수: {argument_strategies['persuasion_metric'].mean():.2f}")

print()

# === 3. 개별 전략별 평균 설득력 점수 ===
print("3. 개별 전략별 평균 설득력 점수")
print("-" * 40)

individual_strategy_scores = []

for strategy in gemini_df_with_scores['final_strategy'].unique():
    # 해당 전략이 포함된 논증들
    args_with_strategy = argument_strategies[
        argument_strategies['final_strategy'].apply(lambda x: strategy in x)
    ]
    
    if len(args_with_strategy) > 0:
        avg_score = args_with_strategy['persuasion_metric'].mean()
        individual_strategy_scores.append({
            'strategy': strategy,
            'avg_score': avg_score,
            'count': len(args_with_strategy)
        })

individual_df = pd.DataFrame(individual_strategy_scores).sort_values('avg_score', ascending=False)

print("개별 전략별 평균 설득력 점수 (높은 순):")
for _, row in individual_df.iterrows():
    print(f"  {row['strategy']}: {row['avg_score']:.3f} (논증 {row['count']}개)")

print()

# === 4. 2개 전략 조합별 평균 설득력 점수 ===
print("4. 2개 전략 조합별 평균 설득력 점수")
print("-" * 40)

pair_scores = []

for _, row in argument_strategies.iterrows():
    strategies = row['final_strategy']
    score = row['persuasion_metric']
    
    if len(strategies) >= 2:
        for pair in combinations(sorted(strategies), 2):
            pair_scores.append({
                'combination': pair,
                'score': score,
                'argument_id': row['argument_id']
            })

if pair_scores:
    pair_df = pd.DataFrame(pair_scores)
    pair_avg_scores = pair_df.groupby('combination').agg({
        'score': ['mean', 'count']
    }).reset_index()
    pair_avg_scores.columns = ['combination', 'avg_score', 'count']
    
    # 최소 2개 논증에서 나타난 조합만 분석
    pair_avg_scores = pair_avg_scores[pair_avg_scores['count'] >= 2]
    pair_avg_scores = pair_avg_scores.sort_values('avg_score', ascending=False)
    
    print("2개 전략 조합별 평균 설득력 점수 (상위 10개):")
    for _, row in pair_avg_scores.head(10).iterrows():
        combo = row['combination']
        print(f"  {combo[0]} + {combo[1]}: {row['avg_score']:.3f} (논증 {row['count']}개)")

print()

# === 5. 3개 전략 조합별 평균 설득력 점수 ===
print("5. 3개 전략 조합별 평균 설득력 점수")
print("-" * 40)

triplet_scores = []

for _, row in argument_strategies.iterrows():
    strategies = row['final_strategy']
    score = row['persuasion_metric']
    
    if len(strategies) >= 3:
        for triplet in combinations(sorted(strategies), 3):
            triplet_scores.append({
                'combination': triplet,
                'score': score,
                'argument_id': row['argument_id']
            })

if triplet_scores:
    triplet_df = pd.DataFrame(triplet_scores)
    triplet_avg_scores = triplet_df.groupby('combination').agg({
        'score': ['mean', 'count']
    }).reset_index()
    triplet_avg_scores.columns = ['combination', 'avg_score', 'count']
    
    # 최소 2개 논증에서 나타난 조합만 분석
    triplet_avg_scores = triplet_avg_scores[triplet_avg_scores['count'] >= 2]
    triplet_avg_scores = triplet_avg_scores.sort_values('avg_score', ascending=False)
    
    print("3개 전략 조합별 평균 설득력 점수 (상위 10개):")
    for _, row in triplet_avg_scores.head(10).iterrows():
        combo = row['combination']
        print(f"  {combo[0]} + {combo[1]} + {combo[2]}: {row['avg_score']:.3f} (논증 {row['count']}개)")

print()

# === 6. 전략 수별 평균 설득력 점수 ===
print("6. 전략 수별 평균 설득력 점수")
print("-" * 40)

strategy_count_scores = argument_strategies.groupby('strategy_count').agg({
    'persuasion_metric': ['mean', 'count', 'std']
}).reset_index()
strategy_count_scores.columns = ['strategy_count', 'avg_score', 'count', 'std_score']

print("전략 수별 평균 설득력 점수:")
for _, row in strategy_count_scores.iterrows():
    std_val = row['std_score'] if pd.notna(row['std_score']) else 0
    print(f"  {row['strategy_count']}개 전략: {row['avg_score']:.3f} ± {std_val:.3f} (논증 {row['count']}개)")

print()

# === 7. 최고 설득력 논증들의 전략 조합 분석 ===
print("7. 최고 설득력 논증들의 전략 조합 분석")
print("-" * 40)

# 설득력 점수 상위 25% 논증들
top_quartile_threshold = argument_strategies['persuasion_metric'].quantile(0.75)
top_arguments = argument_strategies[argument_strategies['persuasion_metric'] >= top_quartile_threshold]

print(f"상위 25% 논증들 (설득력 점수 {top_quartile_threshold:.1f} 이상, {len(top_arguments)}개):")
print()

for _, row in top_arguments.sort_values('persuasion_metric', ascending=False).iterrows():
    strategies = sorted(list(row['final_strategy']))
    print(f"📋 설득력 점수 {row['persuasion_metric']:.1f}: {len(strategies)}개 전략")
    print(f"   전략: {', '.join(strategies)}")
    print(f"   논증 (처음 100자): {row['original_argument'][:100]}...")
    print()

# === 8. 상관관계 분석 ===
print("8. 상관관계 분석")
print("-" * 40)

# 전략 수와 설득력 점수 간의 상관관계
correlation = argument_strategies['strategy_count'].corr(argument_strategies['persuasion_metric'])
print(f"전략 수와 설득력 점수 간 상관계수: {correlation:.3f}")

if abs(correlation) < 0.1:
    correlation_strength = "매우 약함"
elif abs(correlation) < 0.3:
    correlation_strength = "약함"
elif abs(correlation) < 0.5:
    correlation_strength = "보통"
elif abs(correlation) < 0.7:
    correlation_strength = "강함"
else:
    correlation_strength = "매우 강함"

print(f"상관관계 강도: {correlation_strength}")

print()

# === 9. 결과 저장 ===
print("9. 결과 저장")
print("-" * 40)

# 개별 전략 효과성 저장
individual_df.to_csv('individual_strategy_effectiveness.csv', index=False, encoding='utf-8-sig')
print("✓ 개별 전략 효과성: 'individual_strategy_effectiveness.csv'")

# 2개 조합 효과성 저장
if not pair_avg_scores.empty:
    pair_export = pair_avg_scores.copy()
    pair_export['combination'] = pair_export['combination'].apply(lambda x: f"{x[0]} + {x[1]}")
    pair_export.to_csv('pair_strategy_effectiveness.csv', index=False, encoding='utf-8-sig')
    print("✓ 2개 전략 조합 효과성: 'pair_strategy_effectiveness.csv'")

# 3개 조합 효과성 저장
if not triplet_avg_scores.empty:
    triplet_export = triplet_avg_scores.copy()
    triplet_export['combination'] = triplet_export['combination'].apply(lambda x: f"{x[0]} + {x[1]} + {x[2]}")
    triplet_export.to_csv('triplet_strategy_effectiveness.csv', index=False, encoding='utf-8-sig')
    print("✓ 3개 전략 조합 효과성: 'triplet_strategy_effectiveness.csv'")

# 전략 수별 효과성 저장
strategy_count_scores.to_csv('strategy_count_effectiveness.csv', index=False, encoding='utf-8-sig')
print("✓ 전략 수별 효과성: 'strategy_count_effectiveness.csv'")

print()
print("=" * 80)
print("전략 조합별 설득력 효과 분석 완료!")
print("=" * 80) 