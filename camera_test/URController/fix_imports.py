import os
import re

def fix_imports(directory):
    """
    Fix import statements in all Python files in the given directory.
    Replace 'import URController.URBasic' with a relative import setup.
    """
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            filepath = os.path.join(directory, filename)
            print(f"Processing {filepath}")
            
            with open(filepath, 'r') as file:
                content = file.read()
            
            # Replace the problematic import statement
            if 'import URController.URBasic' in content:
                new_content = content.replace(
                    'import URController.URBasic as URBasic',
                    'import sys\nimport os\n# Add parent directory to path to allow imports\nsys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\nimport URBasic'
                )
                
                with open(filepath, 'w') as file:
                    file.write(new_content)
                print(f"Fixed imports in {filepath}")

if __name__ == "__main__":
    # Path to URBasic directory
    urbasic_dir = os.path.join('URBasic')
    fix_imports(urbasic_dir)
    print("Import fixing complete!") 