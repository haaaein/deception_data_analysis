import pandas as pd
import json

print("=" * 80)
print("GEMINI 모델 전략 레이블링 예시 분석")
print("=" * 80)
print()

# 데이터 로드
df = pd.read_csv('deception_strategy_analysis.csv')
gemini_df = df[df['model'] == 'gemini-2.5-pro'].copy()

with open('final_taxonomy_mapping.json', 'r') as f:
    final_taxonomy_mapping = json.load(f)

gemini_df['final_strategy'] = gemini_df['mapped_strategy'].map(final_taxonomy_mapping)

# 논증 ID 생성
if 'argument' in gemini_df.columns:
    gemini_df.rename(columns={'argument': 'original_argument'}, inplace=True)
gemini_df['argument_id'] = pd.factorize(gemini_df['original_argument'])[0]

# 중복 제거
gemini_df_unique = gemini_df.drop_duplicates(subset=['argument_id', 'version', 'final_strategy'])

# 예시 분석을 위한 논증 선택
example_args = [0, 5, 10]  # 첫 번째, 여섯 번째, 열한 번째 논증

print("📋 **Gemini 모델 전략 레이블링 예시 분석**")
print()

for i, arg_id in enumerate(example_args, 1):
    print(f"## 예시 {i}: 논증 ID {arg_id}")
    print("-" * 60)
    
    # 해당 논증의 모든 버전 데이터
    arg_data = gemini_df_unique[gemini_df_unique['argument_id'] == arg_id]
    
    if arg_data.empty:
        print(f"논증 ID {arg_id}에 대한 데이터가 없습니다.")
        continue
    
    # 논증 텍스트 출력 (일부만)
    argument_text = arg_data['original_argument'].iloc[0]
    print(f"**논증 텍스트 (일부):**")
    print(f'"{argument_text[:200]}..."')
    print()
    
    # 버전별 식별된 전략들
    print(f"**버전별 식별된 전략:**")
    for version in sorted(arg_data['version'].unique()):
        version_data = arg_data[arg_data['version'] == version]
        strategies = version_data['final_strategy'].tolist()
        print(f"  {version}: {strategies} ({len(strategies)}개)")
    
    print()
    
    # 전략별 상세 분석
    print(f"**전략별 상세 분석:**")
    strategy_details = arg_data.groupby('final_strategy').agg({
        'version': lambda x: list(x),
        'sub_strategy': lambda x: list(x),
        'mapped_strategy': lambda x: list(x)
    }).reset_index()
    
    for _, row in strategy_details.iterrows():
        strategy = row['final_strategy']
        versions = row['version']
        sub_strategies = row['sub_strategy']
        mapped_strategies = row['mapped_strategy']
        
        print(f"  🎯 **{strategy}**")
        print(f"     - 식별 버전: {versions}")
        print(f"     - 하위 전략: {list(set(sub_strategies))}")
        print(f"     - 매핑된 전략: {list(set(mapped_strategies))}")
        print()
    
    # 일관성 분석
    all_strategies = set()
    version_strategies = {}
    for version in sorted(arg_data['version'].unique()):
        version_data = arg_data[arg_data['version'] == version]
        strategies = set(version_data['final_strategy'].tolist())
        version_strategies[version] = strategies
        all_strategies.update(strategies)
    
    # Jaccard Index 계산
    jaccard_scores = []
    versions = list(version_strategies.keys())
    for i in range(len(versions)):
        for j in range(i+1, len(versions)):
            v1, v2 = versions[i], versions[j]
            intersection = len(version_strategies[v1].intersection(version_strategies[v2]))
            union = len(version_strategies[v1].union(version_strategies[v2]))
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)
    
    avg_jaccard = sum(jaccard_scores) / len(jaccard_scores) if jaccard_scores else 0
    
    print(f"**일관성 분석:**")
    print(f"  - 전체 식별된 전략 수: {len(all_strategies)}개")
    print(f"  - 평균 Jaccard Index: {avg_jaccard:.3f}")
    print(f"  - 완전 일치 여부: {'예' if avg_jaccard == 1.0 else '아니오'}")
    print()
    print("=" * 80)
    print()

# 전략별 분포 요약
print("📊 **전략별 레이블링 분포 요약**")
print("-" * 60)

strategy_counts = gemini_df_unique['final_strategy'].value_counts()
strategy_percentages = (strategy_counts / strategy_counts.sum()) * 100

print("**전략별 레이블링 빈도:**")
for strategy, count in strategy_counts.items():
    percentage = strategy_percentages[strategy]
    print(f"  {strategy}: {count:,}개 ({percentage:.1f}%)")

print()

# 버전별 일관성 통계
print("📈 **버전별 일관성 통계**")
print("-" * 60)

# 각 논증별 Jaccard Index 계산
consistency_results = []
for arg_id in gemini_df_unique['argument_id'].unique():
    arg_data = gemini_df_unique[gemini_df_unique['argument_id'] == arg_id]
    
    version_strategies = {}
    for version in sorted(arg_data['version'].unique()):
        version_data = arg_data[arg_data['version'] == version]
        version_strategies[version] = set(version_data['final_strategy'].values)
    
    versions = list(version_strategies.keys())
    jaccard_scores = []
    for i in range(len(versions)):
        for j in range(i+1, len(versions)):
            v1, v2 = versions[i], versions[j]
            intersection = len(version_strategies[v1].intersection(version_strategies[v2]))
            union = len(version_strategies[v1].union(version_strategies[v2]))
            jaccard = intersection / union if union > 0 else 0
            jaccard_scores.append(jaccard)
    
    if jaccard_scores:
        avg_jaccard = sum(jaccard_scores) / len(jaccard_scores)
        consistency_results.append({
            'argument_id': arg_id,
            'avg_jaccard': avg_jaccard,
            'strategy_count': len(set(arg_data['final_strategy']))
        })

consistency_df = pd.DataFrame(consistency_results)

print(f"**전체 논증 일관성 통계:**")
print(f"  - 평균 Jaccard Index: {consistency_df['avg_jaccard'].mean():.3f}")
print(f"  - 표준편차: {consistency_df['avg_jaccard'].std():.3f}")
print(f"  - 최고 일치도: {consistency_df['avg_jaccard'].max():.3f}")
print(f"  - 최저 일치도: {consistency_df['avg_jaccard'].min():.3f}")

# 일치도 구간별 분포
jaccard_bins = pd.cut(consistency_df['avg_jaccard'], 
                     bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0], 
                     labels=['매우 낮음', '낮음', '중간', '높음', '매우 높음'])
jaccard_distribution = jaccard_bins.value_counts()

print(f"\n**일치도 구간별 분포:**")
for level, count in jaccard_distribution.items():
    percentage = (count / len(consistency_df)) * 100
    print(f"  {level}: {count}개 논증 ({percentage:.1f}%)")

print()
print("=" * 80)
print("분석 완료")
print("=" * 80) 