#dxflib/DxfLib/parse_text.py
import ezdxf
import re
import os
import sys
from pathlib import Path

def parse_text(text):

    text = text.strip()

    numbers = re.findall(r'(\d+\.\d+)', text)
    if not numbers:
        return 0, 0

    main_value = 0
    tolerance = 0

    if numbers:
        match_plus = re.search(r'(\d+\.\d+)±(\d+\.\d+)', text)
        if match_plus:
            main_value = float(match_plus.group(1))
            tolerance = float(match_plus.group(2))
        else:
            match_placeholder = re.search(r'<>±(\d+\.\d+)', text)
            if match_placeholder:
                tolerance = float(match_placeholder.group(1))
                main_value = 0
            else:
                match_p = re.search(r'(\d+\.\d+)?%%[Pp]?(\d+\.\d+)?', text)
                if match_p:
                    main_value = float(match_p.group(1)) if match_p.group(1) else 0
                    tolerance = float(match_p.group(2)) if match_p.group(2) else 0
                else:
                    main_value = float(numbers[0])
                    if any(keyword in text.upper() for keyword in ["MAX", "MIN", "ALLOWED"]):
                        if "MAX AG LACK IS ALLOWED" in text.upper():
                            tolerance = 0.025

    return main_value, tolerance

def extract_numbers(text):
    """
    Extract numbers from complex text using regular expressions
    """
    pattern = r'(\d+\.\d+)'
    match = re.search(pattern, text)
    if match:
        number = match.group(1) if match.group(1) else "0"
        return float(number)
    else:
        return 0

def extract_line_dim(dim, doc):
    """
    Extract numerical values from dimension geometry when direct text parsing fails
    """
    block = doc.blocks.get(dim.dxf.geometry)
    if block is not None:
        for entity in block:
            if entity.dxftype() in ['TEXT', 'MTEXT']:
                text = entity.dxf.text
                value_0 = extract_numbers(text)
                return value_0
    return 0

def extract_dimension_values(entity, doc):
    """
    Extract dimension values and tolerance values from DIMENSION entity
    """
    text = entity.dxf.text if hasattr(entity.dxf, 'text') else ""
    measurement = entity.dxf.measurement if hasattr(entity.dxf, 'measurement') else None

    value_4, value_5 = parse_text(text)

    if value_4 == 0:
        if measurement is not None:
            value_4 = measurement
        else:
            value_4 = extract_line_dim(entity, doc)

    if value_4 < 0:
        value_4 = 0
    if value_5 < 0:
        value_5 = 0

    value_4 = value_4 * 1000
    value_5 = value_5 * 1000

    return value_4, value_5

def normalize_to_255(value, min_val, max_val):
    """
    Normalize value to integer between [0, 255].

    Args:
        value (float): Original value
        min_val (float): Minimum value
        max_val (float): Maximum value

    Returns:
        int: Normalized integer value
    """
    if max_val == min_val or max_val == 0:
        return 0
    normalized = ((value - min_val) / (max_val - min_val)) * 255
    return int(round(max(0, min(255, normalized))))

def parse_dxf_dimensions(file_path):
    """
    Parse all DIMENSION entities in DXF file and extract their dimension values and tolerance values
    """
    try:
        doc = ezdxf.readfile(file_path)
    except Exception as e:
        print(f"Unable to read DXF file {file_path}: {e}")
        return []

    msp = doc.modelspace()

    dimensions = []

    for i, entity in enumerate(msp):
        if entity.dxftype() == 'DIMENSION':
            try:
                dim_value, tolerance = extract_dimension_values(entity, doc)

                dim_info = {
                    'id': i,
                    'dim_value': dim_value,
                    'tolerance': tolerance,
                    'raw_text': entity.dxf.text if hasattr(entity.dxf, 'text') else ""
                }

                dimensions.append(dim_info)
            except Exception as e:
                print(f"Error processing DIMENSION entity #{i}: {e}")

    return dimensions

def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_text.py <dxf_file_or_directory_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    path = Path(input_path)

    if path.is_dir():
        for file_path in path.glob('*.dxf'):
            process_file(file_path)
    elif path.is_file() and path.suffix.lower() == '.dxf':
        process_file(path)
    else:
        print(f"Error: {input_path} is not a valid DXF file or directory")
        sys.exit(1)

def process_file(file_path):
    """Process single DXF file and output results"""
    print(f"\nProcessing file: {file_path}")
    dimensions = parse_dxf_dimensions(file_path)

    if not dimensions:
        print("No DIMENSION entities found or errors occurred during parsing.")
        return

    dim_values = [dim['dim_value'] / 1000 for dim in dimensions]
    tol_values = [dim['tolerance'] / 1000 for dim in dimensions]

    dim_min = min(dim_values) if dim_values and max(dim_values) > 0 else 0.0
    dim_max = max(dim_values) if dim_values and max(dim_values) > 0 else 1.0
    tol_min = min(tol_values) if tol_values and max(tol_values) > 0 else 0.0
    tol_max = max(tol_values) if tol_values and max(tol_values) > 0 else 1.0

    print(f"Found {len(dimensions)} DIMENSION entities:")
    print("-" * 120)
    print(f"{'ID':^5} | {'Dim Value (mm)':^15} | {'Tolerance (mm)':^15} | {'Dim Normalized':^15} | {'Tol Normalized':^15} | {'Raw Text'}")
    print("-" * 120)

    for dim in dimensions:
        normalized_dim = normalize_to_255(dim['dim_value'] / 1000, dim_min, dim_max)
        normalized_tol = normalize_to_255(dim['tolerance'] / 1000, tol_min, tol_max)

        print(f"{dim['id']:^5} | {dim['dim_value']/1000:^15.5f} | {dim['tolerance']/1000:^15.5f} | {normalized_dim:^15d} | {normalized_tol:^15d} | {dim['raw_text']}")

if __name__ == '__main__':
    main()