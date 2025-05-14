"""
File: scripts/concatenate_src.py
Script to concatenate all Python files from src directory into a single text file.
"""

import glob

def concatenate_python_files():
    # Output file
    output_file = "tests_all_files.txt"
    
    # Get all Python files in src directory and subdirectories
    python_files = glob.glob("tests/**/*.py", recursive=True)
    
    # Sort files to ensure consistent order
    python_files.sort()
    
    with open(output_file, "w", encoding="utf-8") as outfile:
        for file_path in python_files:
            # Write file path as a separator
            outfile.write(f"\n{'='*80}\n")
            outfile.write(f"FILE: {file_path}\n")
            outfile.write(f"{'='*80}\n\n")
            
            # Read and write the content of each file
            try:
                with open(file_path, "r", encoding="utf-8") as infile:
                    content = infile.read()
                    outfile.write(content)
                    outfile.write("\n\n")  # Add extra newlines between files
            except Exception as e:
                outfile.write(f"Error reading file: {str(e)}\n")
    
    print(f"All Python files from src directory have been concatenated into {output_file}")
    print(f"Total files processed: {len(python_files)}")

if __name__ == "__main__":
    concatenate_python_files() 