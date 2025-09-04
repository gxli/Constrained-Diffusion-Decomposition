#!/usr/bin/env python3
import json
import base64
import os
from pathlib import Path

def extract_images_from_ipynb(ipynb_path):
    """
    Extract images from a Jupyter Notebook file and save them in the current directory.
    
    Args:
        ipynb_path (str): Path to the .ipynb file
    """
    # Check if file exists and is a .ipynb file
    if not os.path.exists(ipynb_path) or not ipynb_path.endswith('.ipynb'):
        print(f"Error: {ipynb_path} is not a valid .ipynb file")
        return

    try:
        # Read the notebook
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)

        # Counter for naming images
        image_count = 1
        
        # Create output directory if it doesn't exist
        output_dir = Path.cwd()
        
        # Iterate through all cells
        for cell in notebook.get('cells', []):
            # Check for outputs in code cells
            if cell.get('cell_type') == 'code':
                for output in cell.get('outputs', []):
                    # Look for image outputs
                    if 'data' in output:
                        for mime_type, data in output['data'].items():
                            if mime_type.startswith('image/'):
                                # Get the image format (e.g., png, jpeg)
                                image_format = mime_type.split('/')[-1]
                                
                                # Decode base64 image data
                                try:
                                    image_data = base64.b64decode(data)
                                    
                                    # Generate output filename
                                    output_filename = f"notebook_image_{image_count}.{image_format}"
                                    output_path = output_dir / output_filename
                                    
                                    # Save the image
                                    with open(output_path, 'wb') as img_file:
                                        img_file.write(image_data)
                                    
                                    print(f"Saved image: {output_path}")
                                    image_count += 1
                                    
                                except Exception as e:
                                    print(f"Error decoding image {image_count}: {str(e)}")
            
            # Check for markdown cells with embedded images
            elif cell.get('cell_type') == 'markdown':
                source = ''.join(cell.get('source', []))
                # Simple check for base64 encoded images in markdown
                if 'data:image' in source:
                    print(f"Warning: Found potential embedded image in markdown cell. "
                          f"Manual extraction may be needed for cell: {source[:50]}...")

    except Exception as e:
        print(f"Error processing notebook: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python extract_ipynb_images.py <path_to_notebook.ipynb>")
        sys.exit(1)
    
    ipynb_file = sys.argv[1]
    extract_images_from_ipynb(ipynb_file)