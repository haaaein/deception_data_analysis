import pandas as pd
import json

# --- 1. Load and Prepare Data ---

try:
    df = pd.read_csv('deception_strategy_analysis.csv')
    with open('final_taxonomy_mapping.json', 'r') as f:
        final_taxonomy_mapping = json.load(f)
except FileNotFoundError as e:
    print(f"Error: Could not find a required file. {e}")
    exit()

# Map to final 8 strategies
df['final_strategy'] = df['mapped_strategy'].map(final_taxonomy_mapping)
df.dropna(subset=['final_strategy'], inplace=True)

# Create a unique ID for each argument
if 'argument' in df.columns:
    df.rename(columns={'argument': 'original_argument'}, inplace=True)
df['argument_id'] = pd.factorize(df['original_argument'])[0]


# --- 2. Calculate Base Consensus Counts (CORRECTED LOGIC) ---

# **CRITICAL FIX**: Ensure one vote per version per strategy per argument.
df_unique_votes = df.drop_duplicates(subset=['argument_id', 'worker_id', 'final_strategy'])

# This is the most expensive operation, so we do it only once.
# For each argument and strategy, count how many versions (workers) agreed.
consensus_counts = df_unique_votes.groupby(['argument_id', 'final_strategy'])['worker_id'].count().reset_index()
consensus_counts.rename(columns={'worker_id': 'agreement_count'}, inplace=True)


# --- 3. Perform Sensitivity Analysis ---

analysis_results = []

# We test a range of thresholds, from 2 to 10
for threshold in range(2, 11):
    # Filter for strategies that meet the current threshold
    core_strategies_df = consensus_counts[consensus_counts['agreement_count'] >= threshold]

    if core_strategies_df.empty:
        # If no strategies meet the threshold, record zeros and stop early
        analysis_results.append({
            'Threshold': f'>= {threshold}',
            'Total Core Strategies': 0,
            'Arguments with Core Strategies': 0,
            'Avg Strategies per Argument': 0.0
        })
        continue

    # Total number of unique strategies found at this threshold
    total_core_strategies = len(core_strategies_df)

    # Number of unique arguments that have at least one core strategy
    arguments_with_core_strategies = core_strategies_df['argument_id'].nunique()

    # Average number of core strategies PER ARGUMENT THAT HAS THEM
    avg_strategies_per_arg = core_strategies_df.groupby('argument_id').size().mean()

    analysis_results.append({
        'Threshold': f'>= {threshold}',
        'Total Core Strategies': total_core_strategies,
        'Arguments with Core Strategies': arguments_with_core_strategies,
        'Avg Strategies per Argument': round(avg_strategies_per_arg, 2)
    })

# --- 4. Print Results Table ---

results_df = pd.DataFrame(analysis_results)

print("\n--- Consensus Threshold Sensitivity Analysis ---")
print("This table shows how the number of 'core strategies' decreases as we require more versions to agree.\n")
print(results_df.to_string(index=False))
print("\n" + "="*80)
print("[Analysis]")
print(" - 'Total Core Strategies': The total number of strategy labels that met the threshold across all arguments.")
print(" - 'Arguments with Core Strategies': How many unique arguments had at least one core strategy.")
print(" - 'Avg Strategies per Argument': For the arguments that had core strategies, their average count.")
print("A sharp drop indicates a lack of strong consensus beyond that threshold.") 