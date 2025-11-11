"""SVG to NumPy Converter

Converts SVG technical drawings to NumPy arrays for processing.
Extracts line segments with coordinates and optional styling information.
"""

from svgpathtools import svg2paths
from xml.dom import minidom
import numpy as np
import os
import glob

# Process all SVG files in the SheetMetal directory
color_count = 0
for filename in glob.glob('SheetMetal/*.svg'):
    lines = []
    
    # Parse SVG file to extract paths and attributes
    paths, attributes = svg2paths(filename)
    
    for k, v in enumerate(attributes):
        # Extract coordinates from path data
        coord = v["d"].split(" ")
        try:
            M, x1, y1, x2, y2, _ = coord
        except: 
            continue  # Skip malformed coordinates
            
        # Check for styling information (currently not used for classification)
        if "style" in v:
            color = 0
            color_count += 1
        else:
            color = 0
            
        # Store line segment: [x1, y1, x2, y2, color_label]
        lines.append([x1, y1, x2, y2, color])
    
    # Convert to numpy array
    lines = np.asarray(lines).astype(np.float)
    
    try:
        # Normalize coordinates to start from origin
        minimum = np.min(lines, axis=0)
        lines[:, 0] = lines[:, 0] - minimum[0]  # Normalize x1
        lines[:, 1] = lines[:, 1] - minimum[1]  # Normalize y1
        lines[:, 2] = lines[:, 2] - minimum[0]  # Normalize x2
        lines[:, 3] = lines[:, 3] - minimum[1]  # Normalize y2
        
        print(f"Processed {filename}: {len(lines)} line segments")
        
        # Save as numpy file
        output_path = os.path.splitext(filename)[0] + '.npy'
        np.save(output_path, lines)
        
    except ValueError:
        print(f"Error processing {filename}: Invalid data")
        continue

print(f"Total files with styling: {color_count}")