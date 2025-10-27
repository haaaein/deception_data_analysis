import pandas as pd
import json
import re
from collections import Counter

# --- 1. Load Data ---

try:
    df = pd.read_csv('deception_strategy_analysis.csv')
except FileNotFoundError:
    print("Error: 'deception_strategy_analysis.csv' not found.")
    exit()

try:
    with open('final_taxonomy_mapping.json', 'r') as f:
        final_taxonomy_mapping = json.load(f)
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

# --- 2. Pre-processing & Merging ---

# Map the intermediate strategies to the final 8 strategies using the CORRECTED mapping file
df['final_strategy'] = df['mapped_strategy'].map(final_taxonomy_mapping)

# Drop rows where mapping failed (e.g., for 'Unmapped' if not in the JSON file)
df.dropna(subset=['final_strategy'], inplace=True)
df = df[df['final_strategy'] != 'No Deception'] # Exclude cases with no deception for this analysis

# Rename the argument column for merging
if 'argument' in df.columns:
    df.rename(columns={'argument': 'original_argument'}, inplace=True)

df_merged = pd.merge(df, persuasion_df, on='original_argument', how='left')
df_merged['argument_id'] = pd.factorize(df_merged['original_argument'])[0]

# --- 3. Union (Set) Analysis ---

# For each argument, get the set of unique strategies mentioned by ANY version
union_strategies_df = df_merged.groupby('argument_id')['final_strategy'].unique().reset_index()
union_strategies_df['strategy_count'] = union_strategies_df['final_strategy'].apply(len)


# --- 4. Calculate and Print Results ---

# 4.1. Average number of unique strategies per argument (Union)
avg_strategies_per_arg = union_strategies_df['strategy_count'].mean()

print("\n--- Union-Based Analysis Results (All Mentioned Strategies) ---\n")
print(f"1. A New Average: An argument is associated with an average of {avg_strategies_per_arg:.2f} unique strategies (across all 15 versions).")
print("-" * 70)


# 4.2. Distribution of All Mentioned Strategies (Overall)
# Explode the list of strategies to count each one
all_strategies = union_strategies_df.explode('final_strategy')
overall_strategy_distribution = all_strategies['final_strategy'].value_counts(normalize=True) * 100

print("2. Overall Distribution of All Mentioned Strategies:")
for strategy, percentage in overall_strategy_distribution.items():
    print(f"   - {strategy}: {percentage:.2f}%")
print("-" * 70)


# 4.3. Analysis of Highly Persuasive Arguments
persuasive_args_df = pd.merge(
    union_strategies_df,
    df_merged[['argument_id', 'persuasion_metric']].drop_duplicates(),
    on='argument_id',
    how='left'
)

highly_persuasive_df = persuasive_args_df[persuasive_args_df['persuasion_metric'] >= 2]

if not highly_persuasive_df.empty:
    persuasive_strategy_list = highly_persuasive_df.explode('final_strategy')
    persuasive_strategy_distribution = persuasive_strategy_list['final_strategy'].value_counts(normalize=True) * 100
    print("3. Strategy Distribution in Highly Persuasive Arguments (Metric >= 2):")
    for strategy, percentage in persuasive_strategy_distribution.items():
        print(f"   - {strategy}: {percentage:.2f}%")
else:
    print("3. No arguments with a persuasion metric >= 2 were found in the dataset.")
print("-" * 70)

# 4.4. Save the results for further inspection
union_strategies_df.to_csv('union_strategy_analysis.csv', index=False)
print("4. Saved the detailed union analysis to 'union_strategy_analysis.csv'") 