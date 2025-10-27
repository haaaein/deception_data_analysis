import pandas as pd
import json

# Set display options to show full content
pd.set_option('display.max_rows', None)
pd.set_option('display.max_colwidth', None)

# --- 1. Load Data ---

try:
    df = pd.read_csv('deception_strategy_analysis.csv')
except FileNotFoundError:
    print("Error: 'deception_strategy_analysis.csv' not found. Please run 'analyze_deception_strategies.py' first.")
    exit()

try:
    with open('final_taxonomy_mapping.json', 'r') as f:
        final_taxonomy_mapping = json.load(f)
except FileNotFoundError:
    print("Error: 'final_taxonomy_mapping.json' not found.")
    exit()


# --- 2. Select and Display a Single Argument ---

if df.empty:
    print("The analysis file 'deception_strategy_analysis.csv' is empty.")
    exit()

# Select the first argument as the sample
sample_argument_text = df['argument'].iloc[0]

# Filter the DataFrame for this specific argument
arg_df = df[df['argument'] == sample_argument_text].copy()

# Apply the final taxonomy mapping
# We use .get() to avoid errors if a mapped_strategy is not in our final mapping
arg_df['final_strategy'] = arg_df['mapped_strategy'].apply(lambda x: final_taxonomy_mapping.get(x))


# --- 3. Print the Detailed Breakdown ---

print("--- Debugging Analysis for a Single Argument ---")
print("\n[ARGUMENT TEXT]")
print(sample_argument_text)
print("\n" + "="*80)
print("\n[STRATEGIES DETECTED BY 15 VERSIONS]")

# Check if the filtered dataframe is empty
if arg_df.empty:
    print("Could not find any strategy entries for the selected argument.")
else:
    # Select and reorder columns for clear presentation
    output_df = arg_df[['model', 'version', 'sub_strategy', 'mapped_strategy', 'final_strategy']]
    print(output_df.to_string())

print("\n" + "="*80)
print("\n[ANALYSIS]")
print("The table above shows how a single argument was interpreted by each of the 15 models/versions.")
print("Please review this breakdown. The lack of consensus likely stems from:")
print("1. Highly varied 'sub_strategy' labels from the initial open coding.")
print("2. Different intermediate 'mapped_strategy' categories from each version's unique taxonomy.")
print("3. These differences persisting even after mapping to the 'final_strategy'.") 