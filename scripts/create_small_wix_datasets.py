import json
import os
from pathlib import Path

def create_small_dataset(input_path, output_path, limit=50):
    """
    Read a JSON file (list of objects), take the first `limit` items,
    and write them to a new file.
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            print(f"Warning: {input_path} is not a list. Skipping.")
            return

        small_data = data[:limit]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(small_data, f, indent=2, ensure_ascii=False)
            
        print(f"Successfully created {output_path} with {len(small_data)} entries.")
        
    except FileNotFoundError:
        print(f"Error: {input_path} not found.")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

def main():
    base_dir = Path("/localhome/local-jilei/NeMo-Agent-Toolkit/my_data")
    files_to_process = [
        "wixqa_expertwritten.json",
        "wixqa_simulated.json",
        "wixqa_synthetic.json"
    ]

    for filename in files_to_process:
        input_path = base_dir / filename
        # Create output filename: name.json -> name_small.json
        output_filename = filename.replace(".json", "_small.json")
        output_path = base_dir / output_filename
        
        create_small_dataset(input_path, output_path)

if __name__ == "__main__":
    main()

