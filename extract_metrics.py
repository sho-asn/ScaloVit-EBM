import json
import os
import argparse
import re

def extract_metrics_from_folder(folder_path):
    """
    Extracts F1 Score and ROC_AUC from detection_metrics.json in a given folder.

    Args:
        folder_path (str): The path to the folder containing detection_metrics.json.
    
    Returns:
        dict: A dictionary containing the extracted metrics.
    """
    json_file_path = os.path.join(folder_path, 'detection_metrics.json')
    if not os.path.exists(json_file_path):
        print(f"detection_metrics.json not found in {folder_path}")
        return None

    with open(json_file_path, 'r') as f:
        data = json.load(f)

    extracted_data = {}
    for key, value in data.items():
        if key.startswith('test_FaultyCase'):
            # Normalize the key
            match = re.match(r'test_(FaultyCase\d+_Set\d+_\d+)', key, re.IGNORECASE)
            if match:
                normalized_key = match.group(1).lower()
                if isinstance(value, dict) and 'F1 Score' in value and 'ROC_AUC' in value:
                    extracted_data[normalized_key] = {
                        'F1 Score': value['F1 Score'],
                        'ROC_AUC': value['ROC_AUC']
                    }
        elif key == 'overall':
            if isinstance(value, dict) and 'F1 Score' in value and 'ROC_AUC' in value:
                extracted_data['overall'] = {
                    'F1 Score': value['F1 Score'],
                    'ROC_AUC': value['ROC_AUC']
                }
    return extracted_data

def main():
    """
    Main function to parse arguments and call the extraction function.
    """
    parser = argparse.ArgumentParser(description='Extract F1 Score and ROC_AUC from detection_metrics.json.')
    parser.add_argument('folders', nargs='+', help='List of folder names under the results directory.')
    args = parser.parse_args()

    all_metrics = {}
    results_dir = 'results'

    for folder in args.folders:
        folder_path = os.path.join(results_dir, folder)
        if os.path.isdir(folder_path):
            metrics = extract_metrics_from_folder(folder_path)
            if metrics:
                all_metrics[folder] = metrics
        else:
            print(f"Folder not found: {folder_path}")

    output_file_path = os.path.join(results_dir, 'combined_metrics.json')
    with open(output_file_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"All extracted metrics saved to {output_file_path}")

if __name__ == '__main__':
    main()