import pandas as pd
import json
import re
from collections import Counter

# --- 1. Load Data ---

# Load the main analysis file
try:
    df = pd.read_csv('deception_strategy_analysis.csv')
except FileNotFoundError:
    print("Error: 'deception_strategy_analysis.csv' not found. Please run the initial analysis script first.")
    exit()

# Load the final taxonomy mapping
try:
    with open('final_taxonomy_mapping.json', 'r') as f:
        final_taxonomy_mapping = json.load(f)
except FileNotFoundError:
    print("Error: 'final_taxonomy_mapping.json' not found. Please create the mapping file.")
    exit()

# Load the persuasion data from matched_sample.json
def parse_json_lines(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    # Use regex to find all JSON objects, assuming they start with '{' and end with '}'
    json_strings = re.findall(r'{.*?}', content, re.DOTALL)
    data = []
    for i, s in enumerate(json_strings):
        try:
            data.append(json.loads(s))
        except json.JSONDecodeError:
            print(f"Warning: Could not decode JSON object number {i+1} in {file_path}. Skipping.")
            continue
    return data

persuasion_data = parse_json_lines('matched_sample.json')
if not persuasion_data:
    print("Error: Could not parse any valid JSON from 'matched_sample.json'. Exiting.")
    exit()

persuasion_df = pd.DataFrame(persuasion_data)

# --- Check for both possible key names for persuasion metric ---
persuasion_metric_key = None
if 'persuasion_metric' in persuasion_df.columns:
    persuasion_metric_key = 'persuasion_metric'
elif 'persuasiveness_metric' in persuasion_df.columns:
    persuasion_metric_key = 'persuasiveness_metric'

if 'argument' not in persuasion_df.columns or not persuasion_metric_key:
    print("Error: 'matched_sample.json' must contain 'argument' and either 'persuasion_metric' or 'persuasiveness_metric' keys in its objects.")
    exit()

# Rename columns for consistency
persuasion_df.rename(columns={
    'argument': 'original_argument',
    persuasion_metric_key: 'persuasion_metric'
}, inplace=True)

persuasion_df = persuasion_df[['original_argument', 'persuasion_metric']]


# --- 2. Pre-processing ---

# Map the intermediate strategies to the final 8 strategies
df['final_strategy'] = df['mapped_strategy'].map(final_taxonomy_mapping)

# Drop rows where the strategy couldn't be mapped (if any)
df.dropna(subset=['final_strategy'], inplace=True)

# Ensure the merge key column 'argument' is renamed to 'original_argument' for consistency
if 'argument' in df.columns:
    df.rename(columns={'argument': 'original_argument'}, inplace=True)
else:
    print("Error: The main analysis DataFrame must contain an 'argument' column.")
    exit()

# Merge with persuasion data
df_merged = pd.merge(df, persuasion_df, on='original_argument', how='left')

# --- 3. Consensus Analysis (CORRECTED LOGIC) ---

# Create a unique ID for each argument for stable grouping
df_merged['argument_id'] = pd.factorize(df_merged['original_argument'])[0]

# **CRITICAL FIX**: A single version should only get one vote per strategy on a given argument.
# We remove duplicates where the same worker_id finds multiple sub-strategies
# that map to the same final_strategy for the same argument.
df_unique_votes = df_merged.drop_duplicates(subset=['argument_id', 'worker_id', 'final_strategy'])


CONSENSUS_THRESHOLD = 3

# Now, count the unique versions (workers) that agree on a strategy for an argument.
# After dropping duplicates, a simple count is sufficient and correct.
consensus_counts = df_unique_votes.groupby(['argument_id', 'final_strategy'])['worker_id'].count().reset_index()
consensus_counts.rename(columns={'worker_id': 'agreement_count'}, inplace=True)


# Filter for strategies that meet the consensus threshold
consensus_strategies_df = consensus_counts[consensus_counts['agreement_count'] >= CONSENSUS_THRESHOLD]

# --- 4. Calculate and Print Results ---

# 4.1. Average number of consensus strategies per argument
avg_strategies_per_arg = consensus_strategies_df.groupby('argument_id').size().mean()

print("\n--- Consensus-Based Analysis Results (Threshold: >=3) ---\n")
print(f"1. A Realistic Average: An argument uses an average of {avg_strategies_per_arg:.2f} core strategies.")
print("-" * 60)


# 4.2. Distribution of Consensus Strategies (Overall)
overall_strategy_distribution = consensus_strategies_df['final_strategy'].value_counts(normalize=True) * 100

print("2. Overall Core Strategy Distribution:")
for strategy, percentage in overall_strategy_distribution.items():
    print(f"   - {strategy}: {percentage:.2f}%")
print("-" * 60)


# 4.3. Analysis of Highly Persuasive Arguments
persuasive_analysis_df = pd.merge(
    consensus_strategies_df,
    df_merged[['argument_id', 'persuasion_metric']].drop_duplicates(),
    on='argument_id',
    how='left'
)

# Filter for highly persuasive arguments
highly_persuasive_df = persuasive_analysis_df[persuasive_analysis_df['persuasion_metric'] >= 2]

if not highly_persuasive_df.empty:
    persuasive_strategy_distribution = highly_persuasive_df['final_strategy'].value_counts(normalize=True) * 100
    print("3. Core Strategy Distribution in Highly Persuasive Arguments (Metric >= 2):")
    for strategy, percentage in persuasive_strategy_distribution.items():
        print(f"   - {strategy}: {percentage:.2f}%")
else:
    print("3. No arguments with a persuasion metric >= 2 met the consensus threshold.")
print("-" * 60)

# 4.4. Save the consensus results for further inspection
consensus_strategies_df.to_csv('consensus_strategy_analysis.csv', index=False)
print("4. Saved the detailed consensus analysis to 'consensus_strategy_analysis.csv'") 