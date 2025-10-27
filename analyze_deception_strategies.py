import os
import json
import pandas as pd
import glob
from collections import defaultdict

def find_taxonomy_files(base_path):
    """
    Finds and maps taxonomy files for each model and version.
    Handles inconsistent file naming by sorting.
    """
    taxonomy_map = defaultdict(dict)
    models = ['deepseek-r1', 'o4-mini', 'gemini-2.5-pro']
    
    for model in models:
        model_path = os.path.join(base_path, model)
        # Handle the .json.json extension issue
        files = sorted(glob.glob(os.path.join(model_path, '*.json*')))
        
        for i, f in enumerate(files):
            # Clean up double extensions if they exist
            if f.endswith('.json.json'):
                clean_f = f[:-5]
                if os.path.exists(clean_f):
                    os.remove(f)
                    f = clean_f
                else:
                    os.rename(f, clean_f)
                    f = clean_f

            version = f"v{i+1}"
            taxonomy_map[model][version] = f
            
    return taxonomy_map

def create_reverse_taxonomy_map(taxonomy_files):
    """
    Creates a reverse lookup map from sub_strategy to main strategy_name.
    Map structure: {model: {version: {sub_strategy: strategy_name}}}
    """
    reverse_map = defaultdict(lambda: defaultdict(dict))
    
    for model, versions in taxonomy_files.items():
        for version, filepath in versions.items():
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                for strategy_group in data:
                    main_strategy = strategy_group.get('strategy_name')
                    sub_strategies = strategy_group.get('sub_strategies', [])
                    if main_strategy:
                        for sub_strategy in sub_strategies:
                            reverse_map[model][version][sub_strategy] = main_strategy
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {filepath}")
            except Exception as e:
                print(f"Warning: Could not process file {filepath}. Error: {e}")
                
    return reverse_map

def process_open_coding_files(base_path, reverse_taxonomy_map):
    """
    Processes all open coding files, maps sub-strategies to main strategies,
    and returns a list of consolidated data.
    Handles different key names for strategies.
    """
    all_data = []
    open_coding_path = os.path.join(base_path, 'open_coding')
    
    version_dirs = glob.glob(os.path.join(open_coding_path, '*', 'v[1-5]'))
    
    for version_dir in version_dirs:
        parts = version_dir.split(os.sep)
        version_name = parts[-1]
        model_name = parts[-2]
        
        coding_files = glob.glob(os.path.join(version_dir, '*.json'))
        
        for coding_file in coding_files:
            try:
                with open(coding_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                if not isinstance(data, list):
                    # print(f"Warning: Data in {coding_file} is not a list. Skipping.")
                    continue

                for item in data:
                    if not isinstance(item, dict):
                        continue

                    worker_id = item.get('worker_id')
                    sub_strategy = item.get('strategy_name') or item.get('final_strategy')
                    argument = item.get('argument')

                    if not worker_id or not sub_strategy:
                        continue
                        
                    taxonomy_lookup = reverse_taxonomy_map.get(model_name, {}).get(version_name, {})
                    mapped_strategy = taxonomy_lookup.get(sub_strategy, 'Unmapped')
                    
                    all_data.append({
                        'worker_id': worker_id,
                        'model': model_name,
                        'version': version_name,
                        'sub_strategy': sub_strategy,
                        'mapped_strategy': mapped_strategy,
                        'argument': argument
                    })

            except json.JSONDecodeError:
                # print(f"Warning: Could not decode JSON from {coding_file}")
                pass
            except Exception as e:
                print(f"Warning: Could not process file {coding_file}. Error: {e}")
                
    return all_data

def apply_final_mapping(df, mapping_filepath):
    """
    Applies the final 8-category mapping to the DataFrame.
    """
    try:
        with open(mapping_filepath, 'r', encoding='utf-8') as f:
            final_mapping_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: Mapping file not found at {mapping_filepath}")
        return df
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {mapping_filepath}")
        return df

    # Create a reverse map for final strategy lookup
    final_map = {}
    for category in final_mapping_data.get('strategy_mapping', []):
        final_strategy_name = category.get('final_strategy_name')
        for strategy in category.get('mapped_strategies', []):
            final_map[strategy] = final_strategy_name
            
    # Add a new column with the final 8 strategies
    df['final_strategy'] = df['mapped_strategy'].map(final_map).fillna('Unmapped')
    
    return df

def main():
    """
    Main function to run the analysis pipeline.
    """
    base_path = '.'
    
    # Check if the first-level analysis file already exists
    first_level_output = 'deception_strategy_analysis.csv'
    if not os.path.exists(first_level_output):
        print("Step 1: Finding all taxonomy files...")
        taxonomy_files = find_taxonomy_files(base_path)
        print(f"Found taxonomy files for {len(taxonomy_files)} models.")
        
        print("\nStep 2: Creating reverse taxonomy map for strategy lookup...")
        reverse_taxonomy_map = create_reverse_taxonomy_map(taxonomy_files)
        
        print("\nStep 3: Processing open coding files and mapping strategies...")
        consolidated_data = process_open_coding_files(base_path, reverse_taxonomy_map)
        print(f"Processed a total of {len(consolidated_data)} entries.")
        
        if not consolidated_data:
            print("No data was processed. Exiting.")
            return
            
        print("\nStep 4: Creating DataFrame and saving to CSV...")
        df = pd.DataFrame(consolidated_data)
        df = df[['worker_id', 'model', 'version', 'sub_strategy', 'mapped_strategy', 'argument']]
        df.to_csv(first_level_output, index=False, encoding='utf-8-sig')
        print(f"Intermediate analysis saved to '{first_level_output}'")
    else:
        print(f"Found existing analysis file: '{first_level_output}'. Loading it.")
        df = pd.read_csv(first_level_output)

    # --- Final Analysis ---
    print("\n--- Final 8-Category Mapping ---")
    mapping_filepath = 'final_taxonomy_mapping.json'
    df_final = apply_final_mapping(df, mapping_filepath)
    
    final_output_filename = 'final_deception_analysis.csv'
    df_final.to_csv(final_output_filename, index=False, encoding='utf-8-sig')
    print(f"Final analysis complete. Results saved to '{final_output_filename}'")
    
    print("\n--- Final Analysis Summary ---")
    print("Distribution of Final 8 Strategies (Overall):")
    print(df_final['final_strategy'].value_counts())
    
    print("\nDistribution of Final 8 Strategies (per Model):")
    print(df_final.groupby('model')['final_strategy'].value_counts())

if __name__ == '__main__':
    main() 