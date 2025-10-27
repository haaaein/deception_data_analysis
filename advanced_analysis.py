import pandas as pd
import json
import re

def load_data(deception_file, persuasion_file):
    """Loads and merges the two main data files, handling potential JSON errors robustly."""
    try:
        df_deception = pd.read_csv(deception_file)
        
        persuasion_data = []
        with open(persuasion_file, 'r', encoding='utf-8') as f:
            content = f.read()
            json_objects = re.findall(r'\{.*?\}', content, re.DOTALL)
            for obj_str in json_objects:
                try:
                    if obj_str.count('{') == obj_str.count('}'):
                         persuasion_data.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    pass

        df_persuasion = pd.DataFrame(persuasion_data)
        
        # 필요한 컬럼만 선택하고 중복 제거
        if 'worker_id' in df_persuasion.columns:
            df_persuasion = df_persuasion[['worker_id', 'persuasiveness_metric', 'argument']].drop_duplicates(subset=['worker_id'])
        else:
            print("Warning: 'worker_id' not found in persuasion data.")
            return df_deception # Or handle as an error

        # Merge the dataframes
        df_merged = pd.merge(df_deception, df_persuasion, on='worker_id', how='left', suffixes=('_deception', '_persuasion'))
        
        # Safely combine 'argument' columns
        if 'argument_persuasion' in df_merged.columns:
            df_merged['argument'] = df_merged['argument_persuasion'].fillna(df_merged['argument_deception'])
        else: # If no conflict, the column is just 'argument' or 'argument_deception'
            df_merged.rename(columns={'argument_deception': 'argument'}, inplace=True)

        # Drop the redundant columns
        df_merged.drop(columns=['argument_deception', 'argument_persuasion'], inplace=True, errors='ignore')

        return df_merged
        
    except FileNotFoundError as e:
        print(f"Error: {e}. Please make sure both CSV files are in the correct directory.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during data loading: {e}")
        return None

def analyze_argument_strategies(df):
    """
    Analyzes strategies per argument.
    1. Gets the union of strategies for each argument.
    2. Calculates the average number of strategies used per argument.
    """
    if df is None:
        return None, 0
        
    df_filtered = df[df['final_strategy'] != 'Unmapped']
    df_filtered = df_filtered.dropna(subset=['argument', 'final_strategy'])

    if df_filtered.empty:
        return None, 0
        
    argument_strategies = df_filtered.groupby(['worker_id', 'argument'])['final_strategy'].agg(set).reset_index()
    argument_strategies.rename(columns={'final_strategy': 'unique_strategies'}, inplace=True)
    
    argument_strategies['num_strategies'] = argument_strategies['unique_strategies'].apply(len)
    avg_strategies_per_argument = argument_strategies['num_strategies'].mean()
    
    return argument_strategies, avg_strategies_per_argument

def analyze_persuasive_arguments(df):
    """
    Analyzes strategies used in highly persuasive arguments (persuasion_metric >= 2).
    """
    if df is None or 'persuasiveness_metric' not in df.columns:
        print("Warning: 'persuasiveness_metric' column not found.")
        return None, None
        
    persuasive_df = df[df['persuasiveness_metric'] >= 2].copy()
    
    if persuasive_df.empty:
        print("No arguments found with persuasion_metric >= 2.")
        return None, None
        
    persuasive_strategy_dist = persuasive_df['final_strategy'].value_counts()
    
    # Aggregate strategies for each persuasive argument
    persuasive_agg = persuasive_df.groupby(['worker_id', 'argument', 'persuasiveness_metric'])['final_strategy'].agg(set).reset_index()
    persuasive_agg.rename(columns={'final_strategy': 'unique_strategies'}, inplace=True)
    
    return persuasive_strategy_dist, persuasive_agg

def main():
    """Main function to run the advanced analysis."""
    deception_file = 'final_deception_analysis.csv'
    persuasion_file = 'matched_sample.json'
    
    print("Step 1: Loading and merging data...")
    df_merged = load_data(deception_file, persuasion_file)
    
    if df_merged is None:
        print("Data loading failed. Exiting.")
        return

    print("\nStep 2: Analyzing strategies per argument...")
    argument_strategies_df, avg_strategies = analyze_argument_strategies(df_merged)
    
    if argument_strategies_df is not None:
        print(f"\n--- Analysis 1: Strategies per Argument ---")
        print(f"An argument uses an average of {avg_strategies:.2f} unique deceptive strategies.")
        argument_strategies_df.to_csv('argument_strategy_union.csv', index=False, encoding='utf-8-sig')
        print("Detailed strategy union per argument saved to 'argument_strategy_union.csv'")

    print("\nStep 3: Analyzing strategies in highly persuasive arguments...")
    persuasive_dist, persuasive_agg_df = analyze_persuasive_arguments(df_merged)
    
    if persuasive_dist is not None and not persuasive_dist.empty:
        print("\n--- Analysis 2: Strategies in Highly Persuasive Arguments (persuasion_metric >= 2) ---")
        print("Distribution of strategies in highly persuasive arguments:")
        print(persuasive_dist)
        
        if persuasive_agg_df is not None:
            persuasive_agg_df.to_csv('persuasive_argument_strategies.csv', index=False, encoding='utf-8-sig')
            print("\nStrategies for each highly persuasive argument saved to 'persuasive_argument_strategies.csv'")

if __name__ == "__main__":
    main() 