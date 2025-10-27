import pandas as pd
import json
import re
from collections import Counter

print("=== 기만적 설득 전략 분석: 합집합 접근법 (Union Analysis) ===")
print("목적: 15개 LLM 버전이 전체적으로 식별한 모든 기만적 전략의 패턴 분석")
print()

# --- 1. 데이터 로드 ---
try:
    df = pd.read_csv('deception_strategy_analysis.csv')
    print(f"원본 데이터 로드 완료: {len(df):,}개 레이블링 결과")
except FileNotFoundError:
    print("Error: 'deception_strategy_analysis.csv' not found.")
    exit()

try:
    with open('final_taxonomy_mapping.json', 'r') as f:
        final_taxonomy_mapping = json.load(f)
    print(f"최종 전략 매핑 로드 완료: {len(final_taxonomy_mapping)}개 전략")
except FileNotFoundError:
    print("Error: 'final_taxonomy_mapping.json' not found.")
    exit()

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

persuasion_metric_key = 'persuasiveness_metric' if 'persuasiveness_metric' in persuasion_df.columns else 'persuasion_metric'
persuasion_df.rename(columns={'argument': 'original_argument', persuasion_metric_key: 'persuasion_metric'}, inplace=True)
persuasion_df = persuasion_df[['original_argument', 'persuasion_metric']]
print(f"설득력 데이터 로드 완료: {len(persuasion_df)}개 논증")
print()

# --- 2. 데이터 전처리 ---
print("=== 데이터 전처리 ===")

# 최종 8개 전략으로 매핑
df['final_strategy'] = df['mapped_strategy'].map(final_taxonomy_mapping)
df.dropna(subset=['final_strategy'], inplace=True)
print(f"매핑 후 데이터: {len(df):,}개 레이블링 결과")

# 논증 컬럼명 통일
if 'argument' in df.columns:
    df.rename(columns={'argument': 'original_argument'}, inplace=True)

# 설득력 데이터와 병합
df_merged = pd.merge(df, persuasion_df, on='original_argument', how='left')
df_merged['argument_id'] = pd.factorize(df_merged['original_argument'])[0]
print(f"설득력 데이터 병합 완료: {len(df_merged):,}개 레이블링 결과")

# **핵심 수정**: 중복 제거 - 같은 버전이 같은 논증에서 같은 전략을 여러 번 언급한 경우
print("\n--- 중복 제거 (같은 버전의 같은 논증-전략 조합) ---")
print(f"중복 제거 전: {len(df_merged):,}개 레이블링")
df_unique = df_merged.drop_duplicates(subset=['argument_id', 'worker_id', 'final_strategy'])
print(f"중복 제거 후: {len(df_unique):,}개 고유 레이블링")
print(f"제거된 중복: {len(df_merged) - len(df_unique):,}개")
print()

# --- 3. 합집합 분석 ---
print("=== 합집합 분석 (Union Analysis) ===")
print("각 논증에 대해 15개 버전 중 ANY 버전이 언급한 모든 전략의 집합")
print()

# 각 논증별로 언급된 모든 고유 전략들의 집합
union_strategies_df = df_unique.groupby('argument_id')['final_strategy'].unique().reset_index()
union_strategies_df['strategy_count'] = union_strategies_df['final_strategy'].apply(len)

# 논증 정보 추가
argument_info = df_unique[['argument_id', 'original_argument', 'persuasion_metric']].drop_duplicates()
union_strategies_df = pd.merge(union_strategies_df, argument_info, on='argument_id', how='left')

# --- 4. 결과 분석 ---
print("=== 분석 결과 ===")

# 4.1. 논증당 평균 전략 수
avg_strategies_per_arg = union_strategies_df['strategy_count'].mean()
print(f"1. 논증당 평균 전략 수: {avg_strategies_per_arg:.2f}개")
print(f"   (15개 버전이 전체적으로 식별한 고유 전략의 평균)")

# 4.2. 전략 수 분포
strategy_count_dist = union_strategies_df['strategy_count'].value_counts().sort_index()
print(f"\n2. 논증별 전략 수 분포:")
for count, freq in strategy_count_dist.items():
    percentage = (freq / len(union_strategies_df)) * 100
    print(f"   {count}개 전략: {freq}개 논증 ({percentage:.1f}%)")

# 4.3. 전체 전략 분포
all_strategies = union_strategies_df.explode('final_strategy')
overall_strategy_distribution = all_strategies['final_strategy'].value_counts()
overall_strategy_percentage = (overall_strategy_distribution / overall_strategy_distribution.sum()) * 100

print(f"\n3. 전체 전략 분포 (합집합 기준):")
print(f"   총 전략 언급 수: {overall_strategy_distribution.sum():,}개")
for strategy, count in overall_strategy_distribution.items():
    percentage = overall_strategy_percentage[strategy]
    print(f"   - {strategy}: {count:,}개 ({percentage:.2f}%)")

# 4.4. 설득력별 분석
print(f"\n4. 설득력별 분석:")
high_persuasion = union_strategies_df[union_strategies_df['persuasion_metric'] >= 2]
low_persuasion = union_strategies_df[union_strategies_df['persuasion_metric'] < 2]

if not high_persuasion.empty:
    high_avg = high_persuasion['strategy_count'].mean()
    print(f"   고설득력 논증 (≥2): {len(high_persuasion)}개, 평균 {high_avg:.2f}개 전략")
    
    high_strategies = high_persuasion.explode('final_strategy')['final_strategy'].value_counts()
    high_percentage = (high_strategies / high_strategies.sum()) * 100
    print(f"   고설득력 논증의 전략 분포:")
    for strategy, count in high_strategies.items():
        percentage = high_percentage[strategy]
        print(f"     - {strategy}: {count}개 ({percentage:.1f}%)")

if not low_persuasion.empty:
    low_avg = low_persuasion['strategy_count'].mean()
    print(f"   저설득력 논증 (<2): {len(low_persuasion)}개, 평균 {low_avg:.2f}개 전략")

# --- 5. 결과 저장 ---
print(f"\n=== 결과 저장 ===")
union_strategies_df.to_csv('corrected_union_analysis.csv', index=False, encoding='utf-8-sig')
print("상세 분석 결과 저장: 'corrected_union_analysis.csv'")

# 전략별 통계 저장
strategy_stats = pd.DataFrame({
    'strategy': overall_strategy_distribution.index,
    'mention_count': overall_strategy_distribution.values,
    'percentage': overall_strategy_percentage.values
})
strategy_stats.to_csv('union_strategy_distribution.csv', index=False, encoding='utf-8-sig')
print("전략별 분포 저장: 'union_strategy_distribution.csv'")

print(f"\n=== 합집합 분석 완료 ===")
print("다음 단계: 합의 분석 (Consensus Analysis)을 통한 신뢰도 검증") 