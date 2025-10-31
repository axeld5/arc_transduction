import json
import re

# Color mapping from names to integers
COLOR_MAP = {
    "Black": 0,
    "Blue": 1,
    "Red": 2,
    "Green": 3,
    "Yellow": 4,
    "Gray": 5,
    "Pink": 6,
    "Orange": 7,
    "Cyan": 8,
    "Maroon": 9
}

def parse_grid(grid_text):
    """Parse a grid from text format to 2D array of integers."""
    lines = grid_text.strip().split('\n')
    grid = []
    for line in lines:
        row = []
        colors = line.strip().split()
        for color in colors:
            if color in COLOR_MAP:
                row.append(COLOR_MAP[color])
            else:
                # Handle unexpected color names
                print(f"Warning: Unknown color '{color}', defaulting to Black (0)")
                row.append(0)
        if row:  # Only add non-empty rows
            grid.append(row)
    return grid

def extract_examples_from_conversation(messages):
    """Extract input-output pairs from the conversation format."""
    # Find the user message with the examples
    user_content = None
    assistant_content = None
    
    for msg in messages:
        if msg['role'] == 'user' and 'Given input-output grid pairs' in msg['content']:
            user_content = msg['content']
        elif msg['role'] == 'assistant' and 'The output grid for the test input grid is:' in msg['content']:
            assistant_content = msg['content']
    
    if not user_content:
        raise ValueError("Could not find user message with examples")
    
    # Parse training examples
    train_examples = []
    
    # Find all "Example N" sections
    example_pattern = r'Example \d+\s*\nInput:\s*\n(.*?)\n\nOutput:\s*\n(.*?)(?=\n\n(?:Example \d+|Here is the input))'
    matches = re.finditer(example_pattern, user_content, re.DOTALL)
    
    for match in matches:
        input_text = match.group(1)
        output_text = match.group(2)
        
        train_examples.append({
            "input": parse_grid(input_text),
            "output": parse_grid(output_text)
        })
    
    # Parse test example input
    test_input_pattern = r'Here is the input grid for the test example:\s*\nInput:\s*\n(.*?)(?=\n\nDirectly provide)'
    test_input_match = re.search(test_input_pattern, user_content, re.DOTALL)
    
    if not test_input_match:
        raise ValueError("Could not find test input")
    
    test_input_text = test_input_match.group(1)
    test_input = parse_grid(test_input_text)
    
    # Parse test example output from assistant response
    test_output = None
    if assistant_content:
        # Extract the grid from the code block
        output_pattern = r'```\s*\n(.*?)\n```'
        output_match = re.search(output_pattern, assistant_content, re.DOTALL)
        
        if output_match:
            test_output_text = output_match.group(1)
            test_output = parse_grid(test_output_text)
    
    # Create test example
    test_example = {
        "input": test_input,
        "output": test_output if test_output else []
    }
    
    return train_examples, [test_example]

def conversation_to_arc_json(conversation_data, task_id="conversation_task"):
    """Convert conversation format to ARC JSON format."""
    messages = conversation_data.get('messages', [])
    
    train_examples, test_examples = extract_examples_from_conversation(messages)
    
    arc_task = {
        "train": train_examples,
        "test": test_examples,
        "task_id": task_id
    }
    
    return arc_task

# Example usage with the provided conversation
if __name__ == "__main__":
    # The conversation data from the user
    conversation = {
        'messages': [
            {
                'content': 'You are a world-class puzzle solver with exceptional pattern recognition skills. Your task is to analyze puzzles, spot patterns, and provide direct solutions.', 
                'role': 'system'
            }, 
            {
                'content': '''Given input-output grid pairs as reference examples, carefully observe the patterns to predict the output grid for new test input. Each pair follows the same transformation rule. Grids are 2D arrays represented as strings, with cells (colors) separated by spaces and rows by newlines.
Here are the input and output grids for the reference examples:
Example 1
Input:
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Gray Gray Gray Black Black Black
Gray Black Black Black Gray Gray Gray Gray Black Black
Gray Gray Gray Black Gray Gray Gray Gray Gray Black
Gray Gray Gray Black Black Gray Gray Gray Gray Black
Gray Black Black Black Black Black Gray Gray Gray Black
Black Black Black Black Black Black Black Black Black Black

Output:
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Blue Blue Blue Black Black Black
Red Black Black Black Blue Blue Blue Blue Black Black
Red Red Red Black Blue Blue Blue Blue Blue Black
Red Red Red Black Black Blue Blue Blue Blue Black
Red Black Black Black Black Black Blue Blue Blue Black
Black Black Black Black Black Black Black Black Black Black


Example 2
Input:
Black Black Black Gray Black Black Black Black Gray Gray
Black Black Gray Gray Gray Black Black Black Gray Gray
Black Gray Gray Gray Gray Gray Black Black Gray Gray
Black Black Gray Gray Gray Black Black Black Gray Gray
Black Black Black Gray Black Black Black Black Black Black
Black Black Black Black Black Gray Gray Gray Gray Gray
Black Black Gray Gray Black Black Gray Gray Gray Black
Black Black Gray Gray Gray Black Black Black Black Black
Black Black Gray Gray Gray Black Black Black Black Black
Black Black Gray Gray Black Black Black Black Black Black

Output:
Black Black Black Blue Black Black Black Black Red Red
Black Black Blue Blue Blue Black Black Black Red Red
Black Blue Blue Blue Blue Blue Black Black Red Red
Black Black Blue Blue Blue Black Black Black Red Red
Black Black Black Blue Black Black Black Black Black Black
Black Black Black Black Black Red Red Red Red Red
Black Black Gray Gray Black Black Red Red Red Black
Black Black Gray Gray Gray Black Black Black Black Black
Black Black Gray Gray Gray Black Black Black Black Black
Black Black Gray Gray Black Black Black Black Black Black


Example 3
Input:
Black Black Black Black Gray Black Black Black Black Black
Black Black Black Black Gray Gray Black Black Black Black
Gray Gray Black Black Gray Black Black Black Black Black
Gray Black Black Black Black Black Black Black Black Black
Gray Gray Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Gray Gray Gray Black Black Black Black Black Black Black
Gray Gray Gray Black Black Black Black Black Black Black

Output:
Black Black Black Black Red Black Black Black Black Black
Black Black Black Black Red Red Black Black Black Black
Gray Gray Black Black Red Black Black Black Black Black
Gray Black Black Black Black Black Black Black Black Black
Gray Gray Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Blue Blue Blue Black Black Black Black Black Black Black
Blue Blue Blue Black Black Black Black Black Black Black


Here is the input grid for the test example:
Input:
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Gray Gray Black Black Black Gray Gray Gray Black Black
Gray Gray Black Black Black Gray Gray Black Black Black
Gray Gray Black Black Black Gray Black Black Black Gray
Gray Gray Black Black Black Gray Black Black Gray Gray
Gray Gray Black Black Black Gray Gray Gray Gray Gray
Black Black Black Black Black Black Black Black Black Black


Directly provide the output grids corresponding to the given test input grids, based on the patterns observed in the reference examples.''', 
                'role': 'user'
            }, 
            {
                'content': '''The output grid for the test input grid is:

```
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Black Black Black Black Black Black Black Black Black Black
Red Red Black Black Black Blue Blue Blue Black Black
Red Red Black Black Black Blue Blue Black Black Black
Red Red Black Black Black Blue Black Black Black Blue
Red Red Black Black Black Blue Black Black Blue Blue
Red Red Black Black Black Blue Blue Blue Blue Blue
Black Black Black Black Black Black Black Black Black Black
```''', 
                'role': 'assistant'
            }
        ]
    }
    
    # Convert to ARC format
    arc_task = conversation_to_arc_json(conversation, task_id="example_task")
    
    # Save to JSON file
    output_file = "conversation_output.json"
    with open(output_file, 'w') as f:
        json.dump(arc_task, f, indent=2)
    
    print(f"Successfully converted conversation to ARC format!")
    print(f"Output saved to: {output_file}")
    print(f"Train examples: {len(arc_task['train'])}")
    print(f"Test examples: {len(arc_task['test'])}")
    
    # Print a preview of the first train example
    if arc_task['train']:
        print(f"\nFirst train example input shape: {len(arc_task['train'][0]['input'])}x{len(arc_task['train'][0]['input'][0])}")
        print(f"First train example output shape: {len(arc_task['train'][0]['output'])}x{len(arc_task['train'][0]['output'][0])}")

