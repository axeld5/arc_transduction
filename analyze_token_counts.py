import json
import tiktoken
from pathlib import Path
import statistics

def count_tokens(text, encoding):
    """Count tokens in text using tiktoken."""
    return len(encoding.encode(text))

def analyze_json_file(file_path, encoding):
    """Analyze token counts for a single JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    token_counts = []
    
    # Handle different JSON structures
    if isinstance(data, list):
        # Array of objects
        for item in data:
            # Count tokens for the entire item (serialized as JSON)
            item_text = json.dumps(item)
            token_count = count_tokens(item_text, encoding)
            token_counts.append(token_count)
    elif isinstance(data, dict):
        # Single object or dict of objects
        # For dict, analyze each entry
        for key, value in data.items():
            item_text = json.dumps(value)
            token_count = count_tokens(item_text, encoding)
            token_counts.append(token_count)
    
    if not token_counts:
        return None
    
    return {
        'mean': statistics.mean(token_counts),
        'median': statistics.median(token_counts),
        'min': min(token_counts),
        'max': max(token_counts),
        'count': len(token_counts),
        'total': sum(token_counts)
    }

def main():
    # Initialize tiktoken with cl100k_base encoding (used by GPT-4)
    encoding = tiktoken.get_encoding("cl100k_base")
    
    generated_data_dir = Path("generated_data")
    
    # Get all JSON files except those with "summary" in the name
    json_files = [
        f for f in generated_data_dir.glob("*.json")
        if "summary" not in f.name.lower()
    ]
    
    print("Token Count Analysis for generated_data JSON files")
    print("=" * 80)
    print(f"Using tiktoken encoding: cl100k_base (GPT-4)")
    print("=" * 80)
    print()
    
    results = {}
    
    for json_file in sorted(json_files):
        print(f"Analyzing: {json_file.name}")
        stats = analyze_json_file(json_file, encoding)
        
        if stats:
            results[json_file.name] = stats
            print(f"  Entries: {stats['count']}")
            print(f"  Mean:    {stats['mean']:.2f} tokens")
            print(f"  Median:  {stats['median']:.2f} tokens")
            print(f"  Min:     {stats['min']} tokens")
            print(f"  Max:     {stats['max']} tokens")
            print(f"  Total:   {stats['total']} tokens")
            print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for filename, stats in results.items():
        print(f"{filename}:")
        print(f"  {stats['count']} entries | "
              f"mean: {stats['mean']:.2f} | "
              f"median: {stats['median']:.2f} | "
              f"min: {stats['min']} | "
              f"max: {stats['max']}")
    
    # Save results to JSON
    output_file = generated_data_dir / "token_count_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print()
    print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    main()

