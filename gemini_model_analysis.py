import pandas as pd
import numpy as np
import json
import re
from collections import Counter
from scipy import stats
from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 80)
print("GEMINI 모델 특성 분석 (Model-Specific Analysis)")
print("=" * 80)
print("목적: Gemini-2.5-Pro 모델의 기만적 전략 식별 패턴 심층 분석")
print()

# --- 1. 데이터 로드 및 전처리 ---
print("1. 데이터 로드 및 전처리")
print("-" * 40)

try:
    df = pd.read_csv('deception_strategy_analysis.csv')
    print(f"✓ 원본 데이터 로드: {len(df):,}개 레이블링")
except FileNotFoundError:
    print("✗ 오류: 'deception_strategy_analysis.csv' 파일을 찾을 수 없습니다.")
    exit()

try:
    with open('final_taxonomy_mapping.json', 'r') as f:
        final_taxonomy_mapping = json.load(f)
    print(f"✓ 최종 전략 매핑 로드: {len(final_taxonomy_mapping)}개 전략")
except FileNotFoundError:
    print("✗ 오류: 'final_taxonomy_mapping.json' 파일을 찾을 수 없습니다.")
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
print(f"✓ 설득력 데이터 로드: {len(persuasion_df)}개 논증")

# Gemini 모델 데이터만 필터링
gemini_df = df[df['model'] == 'gemini-2.5-pro'].copy()
print(f"✓ Gemini 모델 데이터 필터링: {len(gemini_df):,}개 레이블링")

# 최종 전략 매핑
gemini_df['final_strategy'] = gemini_df['mapped_strategy'].map(final_taxonomy_mapping)
gemini_df.dropna(subset=['final_strategy'], inplace=True)
print(f"✓ 최종 전략 매핑 완료: {len(gemini_df):,}개 레이블링")

# 논증 컬럼명 통일 및 설득력 데이터 병합
if 'argument' in gemini_df.columns:
    gemini_df.rename(columns={'argument': 'original_argument'}, inplace=True)

gemini_df = pd.merge(gemini_df, persuasion_df, on='original_argument', how='left')
gemini_df['argument_id'] = pd.factorize(gemini_df['original_argument'])[0]

# **핵심**: 중복 제거 - 같은 버전이 같은 논증에서 같은 전략을 여러 번 언급한 경우
print(f"✓ 중복 제거 전: {len(gemini_df):,}개 레이블링")
gemini_df_unique = gemini_df.drop_duplicates(subset=['argument_id', 'version', 'final_strategy'])
print(f"✓ 중복 제거 후: {len(gemini_df_unique):,}개 고유 레이블링")
print(f"✓ 제거된 중복: {len(gemini_df) - len(gemini_df_unique):,}개 ({((len(gemini_df) - len(gemini_df_unique))/len(gemini_df)*100):.1f}%)")
print()

# --- 2. 기본 통계 분석 ---
print("2. 기본 통계 분석")
print("-" * 40)

total_arguments = gemini_df_unique['argument_id'].nunique()
total_versions = gemini_df_unique['version'].nunique()
total_strategies = gemini_df_unique['final_strategy'].nunique()

print(f"✓ 분석 대상 논증 수: {total_arguments}개")
print(f"✓ 분석 대상 버전 수: {total_versions}개 (v1-v5)")
print(f"✓ 식별된 전략 수: {total_strategies}개")
print()

# --- 3. 전략 식별 패턴 분석 ---
print("3. 전략 식별 패턴 분석")
print("-" * 40)

# 3.1 논증별 전략 수 계산
strategies_per_argument = gemini_df_unique.groupby(['argument_id', 'version'])['final_strategy'].count().reset_index()
strategies_per_argument.rename(columns={'final_strategy': 'strategy_count'}, inplace=True)

# 전체 평균
overall_mean = strategies_per_argument['strategy_count'].mean()
overall_std = strategies_per_argument['strategy_count'].std()
print(f"✓ 논증당 평균 전략 수: {overall_mean:.2f} ± {overall_std:.2f}")

# 버전별 평균
version_stats = strategies_per_argument.groupby('version')['strategy_count'].agg(['mean', 'std', 'count']).round(2)
print(f"\n✓ 버전별 전략 수 통계:")
print(version_stats)

# 버전 간 차이 통계적 검정
version_groups = [strategies_per_argument[strategies_per_argument['version'] == v]['strategy_count'].values 
                 for v in sorted(gemini_df_unique['version'].unique())]
f_stat, p_value = stats.f_oneway(*version_groups)
print(f"\n✓ 버전 간 차이 검정 (ANOVA): F={f_stat:.3f}, p={p_value:.4f}")
if p_value < 0.05:
    print("  → 버전 간 유의미한 차이 존재")
else:
    print("  → 버전 간 유의미한 차이 없음")

# 전략 분포
strategy_distribution = strategies_per_argument['strategy_count'].value_counts().sort_index()
print(f"\n✓ 논증별 전략 수 분포:")
for count, freq in strategy_distribution.items():
    percentage = (freq / len(strategies_per_argument)) * 100
    print(f"  {count}개 전략: {freq}개 사례 ({percentage:.1f}%)")
print()

# --- 4. 버전 간 일관성 분석 ---
print("4. 버전 간 일관성 분석")
print("-" * 40)

def calculate_jaccard_index(set1, set2):
    """두 집합의 Jaccard Index 계산"""
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0

# 각 논증별로 버전 간 일치도 계산
consistency_results = []
arguments = gemini_df_unique['argument_id'].unique()

for arg_id in arguments:
    arg_data = gemini_df_unique[gemini_df_unique['argument_id'] == arg_id]
    
    # 각 버전별 전략 집합
    version_strategies = {}
    for version in sorted(arg_data['version'].unique()):
        version_data = arg_data[arg_data['version'] == version]
        version_strategies[version] = set(version_data['final_strategy'].values)
    
    # 모든 버전 쌍에 대해 Jaccard Index 계산
    versions = list(version_strategies.keys())
    for i, v1 in enumerate(versions):
        for j, v2 in enumerate(versions):
            if i < j:  # 중복 방지
                jaccard = calculate_jaccard_index(version_strategies[v1], version_strategies[v2])
                consistency_results.append({
                    'argument_id': arg_id,
                    'version1': v1,
                    'version2': v2,
                    'jaccard_index': jaccard
                })

consistency_df = pd.DataFrame(consistency_results)

# 전체 평균 일치도
mean_jaccard = consistency_df['jaccard_index'].mean()
std_jaccard = consistency_df['jaccard_index'].std()
print(f"✓ 평균 Jaccard Index: {mean_jaccard:.3f} ± {std_jaccard:.3f}")

# 버전 쌍별 일치도
version_pair_consistency = consistency_df.groupby(['version1', 'version2'])['jaccard_index'].mean().round(3)
print(f"\n✓ 버전 쌍별 평균 일치도:")
for (v1, v2), jaccard in version_pair_consistency.items():
    print(f"  {v1} vs {v2}: {jaccard:.3f}")

# 일치도 분포
jaccard_bins = pd.cut(consistency_df['jaccard_index'], bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                     labels=['매우 낮음(0-0.2)', '낮음(0.2-0.4)', '중간(0.4-0.6)', '높음(0.6-0.8)', '매우 높음(0.8-1.0)'])
jaccard_distribution = jaccard_bins.value_counts()
print(f"\n✓ 일치도 분포:")
for level, count in jaccard_distribution.items():
    percentage = (count / len(consistency_df)) * 100
    print(f"  {level}: {count}개 ({percentage:.1f}%)")

# 완전 일치 vs 완전 불일치 분석
complete_matches = (consistency_df['jaccard_index'] == 1.0).sum()
complete_mismatches = (consistency_df['jaccard_index'] == 0.0).sum()
print(f"\n✓ 완전 일치: {complete_matches}개 ({complete_matches/len(consistency_df)*100:.1f}%)")
print(f"✓ 완전 불일치: {complete_mismatches}개 ({complete_mismatches/len(consistency_df)*100:.1f}%)")
print()

# --- 5. 설득력-전략 관계 분석 ---
print("5. 설득력-전략 관계 분석")
print("-" * 40)

# 설득력 점수 분류 (3분위수 기준)
persuasion_scores = gemini_df_unique.dropna(subset=['persuasion_metric'])['persuasion_metric']
q33, q67 = persuasion_scores.quantile([0.33, 0.67])

def classify_persuasion(score):
    if pd.isna(score):
        return 'Unknown'
    elif score <= q33:
        return 'Low'
    elif score <= q67:
        return 'Medium'
    else:
        return 'High'

gemini_df_unique['persuasion_level'] = gemini_df_unique['persuasion_metric'].apply(classify_persuasion)

print(f"✓ 설득력 분류 기준:")
print(f"  Low: ≤ {q33:.2f}")
print(f"  Medium: {q33:.2f} < score ≤ {q67:.2f}")
print(f"  High: > {q67:.2f}")

# 설득력 수준별 분포
persuasion_dist = gemini_df_unique['persuasion_level'].value_counts()
print(f"\n✓ 설득력 수준별 분포:")
for level, count in persuasion_dist.items():
    percentage = (count / len(gemini_df_unique)) * 100
    print(f"  {level}: {count}개 ({percentage:.1f}%)")

# 설득력 수준별 전략 분포
persuasion_strategy_crosstab = pd.crosstab(gemini_df_unique['persuasion_level'], 
                                          gemini_df_unique['final_strategy'], 
                                          normalize='index') * 100

print(f"\n✓ 설득력 수준별 전략 분포 (%):")
print(persuasion_strategy_crosstab.round(1))

# 카이제곱 검정
contingency_table = pd.crosstab(gemini_df_unique['persuasion_level'], gemini_df_unique['final_strategy'])
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
print(f"\n✓ 카이제곱 검정: χ² = {chi2:.3f}, p = {p_value:.4f}, df = {dof}")
if p_value < 0.05:
    print("  → 설득력 수준별 전략 분포에 유의미한 차이 존재")
else:
    print("  → 설득력 수준별 전략 분포에 유의미한 차이 없음")

# 효과 크기 (Cramér's V)
n = contingency_table.sum().sum()
cramers_v = np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))
print(f"✓ 효과 크기 (Cramér's V): {cramers_v:.3f}")

# 설득력 높은 논증에서 가장 많이 나타나는 전략
high_persuasion_strategies = gemini_df_unique[gemini_df_unique['persuasion_level'] == 'High']['final_strategy'].value_counts(normalize=True) * 100
print(f"\n✓ 고설득력 논증에서 가장 많이 식별되는 전략:")
for strategy, percentage in high_persuasion_strategies.head(5).items():
    print(f"  {strategy}: {percentage:.1f}%")
print()

# --- 6. 결과 저장 ---
print("6. 결과 저장")
print("-" * 40)

# 논증별 전략 수 저장
strategies_per_argument.to_csv('gemini_strategies_per_argument.csv', index=False, encoding='utf-8-sig')
print("✓ 논증별 전략 수: 'gemini_strategies_per_argument.csv'")

# 버전 간 일치도 저장
consistency_df.to_csv('gemini_version_consistency.csv', index=False, encoding='utf-8-sig')
print("✓ 버전 간 일치도: 'gemini_version_consistency.csv'")

# 설득력별 전략 분포 저장
persuasion_strategy_crosstab.to_csv('gemini_persuasion_strategy_distribution.csv', encoding='utf-8-sig')
print("✓ 설득력별 전략 분포: 'gemini_persuasion_strategy_distribution.csv'")

# 종합 요약 저장
summary_stats = {
    'total_arguments': total_arguments,
    'total_versions': total_versions,
    'total_strategies': total_strategies,
    'mean_strategies_per_argument': overall_mean,
    'std_strategies_per_argument': overall_std,
    'mean_jaccard_index': mean_jaccard,
    'std_jaccard_index': std_jaccard,
    'complete_matches_percentage': complete_matches/len(consistency_df)*100,
    'complete_mismatches_percentage': complete_mismatches/len(consistency_df)*100,
    'chi2_statistic': chi2,
    'chi2_p_value': p_value,
    'cramers_v': cramers_v
}

with open('gemini_analysis_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary_stats, f, indent=2, ensure_ascii=False)
print("✓ 종합 요약: 'gemini_analysis_summary.json'")
print()

# --- 7. 최종 결론 ---
print("7. 최종 결론")
print("-" * 40)

print(f"📊 **Gemini 모델 특성 분석 결과**")
print(f"")
print(f"**전략 식별 패턴:**")
print(f"- 논증당 평균 {overall_mean:.2f}개 전략 식별")
print(f"- 버전 간 차이: {'유의함' if p_value < 0.05 else '유의하지 않음'}")
print(f"")
print(f"**일관성 분석:**")
print(f"- 평균 Jaccard Index: {mean_jaccard:.3f} (중간 수준)")
print(f"- 완전 일치: {complete_matches/len(consistency_df)*100:.1f}%")
print(f"- 완전 불일치: {complete_mismatches/len(consistency_df)*100:.1f}%")
print(f"")
print(f"**설득력-전략 관계:**")
print(f"- 설득력 수준별 전략 분포 차이: {'유의함' if p_value < 0.05 else '유의하지 않음'}")
print(f"- 효과 크기: {cramers_v:.3f} ({'작음' if cramers_v < 0.3 else '중간' if cramers_v < 0.5 else '큼'})")
print(f"")
print(f"**연구적 함의:**")
print(f"- Gemini 모델은 {overall_mean:.1f}개 내외의 전략을 일관되게 식별")
print(f"- 버전 간 일치도는 중간 수준으로 완전한 일관성은 부족")
print(f"- 설득력과 전략 분포의 관계는 {'통계적으로 유의함' if p_value < 0.05 else '통계적으로 유의하지 않음'}")

print(f"\n" + "=" * 80)
print("GEMINI 모델 분석 완료")
print("=" * 80) 