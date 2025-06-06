import ezdxf
from ezdxf.math import BoundingBox
import math
import os

def print_dxf_info(doc):
    print("\nDXF File Information:")
    print(f"DXF Version: {doc.dxfversion}")
    print(f"Encoding: {doc.encoding}")

    print("\nLayer List:")
    for layer in doc.layers:
        print(f"- {layer.dxf.name}")

    print("\nBlock Definition List:")
    for block in doc.blocks:
        print(f"- Block Name: {block.name}")
        entity_types = {}
        for entity in block:
            entity_type = entity.dxftype()
            entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
        if entity_types:
            print(f"  Contained Entity Types:")
            for entity_type, count in entity_types.items():
                print(f"    * {entity_type}: {count} entities")

    print("\nModel Space Entities:")
    entity_types = {}
    for entity in doc.modelspace():
        entity_type = entity.dxftype()
        entity_types[entity_type] = entity_types.get(entity_type, 0) + 1
    for entity_type, count in entity_types.items():
        print(f"- {entity_type}: {count} entities")

def get_tablet_bounds_and_refs(dxf_path):
    """Get bounds coordinates and reference objects for all TABLET blocks"""
    doc = ezdxf.readfile(dxf_path)

    print_dxf_info(doc)

    msp = doc.modelspace()
    tablet_data = []

    print("\nTrying to find TABLET block references...")

    all_inserts = list(msp.query('INSERT'))
    print(f"Method 1 - Direct INSERT entity query count: {len(all_inserts)}")

    insert_count = 0
    for entity in msp:
        if entity.dxftype() == 'INSERT':
            insert_count += 1
            print(f"Found block reference: {entity.dxf.name}")
    print(f"Method 2 - Entity traversal INSERT count: {insert_count}")

    tablet_pattern = "TABLET"
    for entity in msp:
        if entity.dxftype() == 'INSERT' and entity.dxf.name.upper() == tablet_pattern:
            print(f"\nFound TABLET block reference:")
            print(f"- Block name: {entity.dxf.name}")
            print(f"- Insertion point: {entity.dxf.insert}")
            print(f"- Scale: ({entity.dxf.xscale}, {entity.dxf.yscale})")
            print(f"- Rotation: {entity.dxf.rotation} degrees")

            try:
                block_def = doc.blocks[entity.dxf.name]
                bbox = BoundingBox()
                for block_entity in block_def:
                    bbox.extend(block_entity.get_bbox())

                corners = []
                min_x, min_y, _ = bbox.extmin
                max_x, max_y, _ = bbox.extmax
                corners_local = [
                    (min_x, min_y),
                    (max_x, min_y),
                    (max_x, max_y),
                    (min_x, max_y)
                ]

                rotation_rad = math.radians(entity.dxf.rotation)
                insertion_point = entity.dxf.insert
                scale_x = entity.dxf.xscale
                scale_y = entity.dxf.yscale

                transformed_corners = []
                for x, y in corners_local:
                    x_scaled = x * scale_x
                    y_scaled = y * scale_y

                    x_rot = x_scaled * math.cos(rotation_rad) - y_scaled * math.sin(rotation_rad)
                    y_rot = x_scaled * math.sin(rotation_rad) + y_scaled * math.cos(rotation_rad)

                    x_final = x_rot + insertion_point[0]
                    y_final = y_rot + insertion_point[1]

                    transformed_corners.append((x_final, y_final))

                tablet_data.append({
                    'bounds': transformed_corners,
                    'handle': entity.dxf.handle
                })

            except Exception as e:
                print(f"Error processing TABLET block: {str(e)}")

    print(f"\nTotal found {len(tablet_data)} TABLET block references")
    return tablet_data

def is_point_in_polygon(point, polygon):
    """Check if point is inside polygon"""
    x, y = point
    n = len(polygon)
    inside = False

    j = n - 1
    for i in range(n):
        if ((polygon[i][1] > y) != (polygon[j][1] > y) and
                (x < (polygon[j][0] - polygon[i][0]) * (y - polygon[i][1]) /
                 (polygon[j][1] - polygon[i][1]) + polygon[i][0])):
            inside = not inside
        j = i

    return inside

def get_entity_insertion_point(entity):
    """Get entity insertion point"""
    if entity.dxftype() == 'INSERT':
        return (entity.dxf.insert[0], entity.dxf.insert[1])
    else:
        try:
            bbox = entity.get_bbox()
            center_x = (bbox.extmin[0] + bbox.extmax[0]) / 2
            center_y = (bbox.extmin[1] + bbox.extmax[1]) / 2
            return (center_x, center_y)
        except:
            return None

def extract_entities_for_tablet(doc, tablet_data, output_path_template):
    """Extract entities for each TABLET area and create new DXF files"""
    msp = doc.modelspace()

    for index, tablet in enumerate(tablet_data):
        new_doc = ezdxf.new('R2018')
        new_msp = new_doc.modelspace()

        copied_block_names = set()

        for entity in msp:
            if entity.dxftype() == 'INSERT' and entity.dxf.name == 'TABLET':
                continue

            insertion_point = get_entity_insertion_point(entity)
            if insertion_point is None:
                continue

            if is_point_in_polygon(insertion_point, tablet['bounds']):
                if entity.dxftype() == 'INSERT':
                    block_name = entity.dxf.name
                    if block_name not in copied_block_names:
                        if block_name in doc.blocks and block_name.lower() not in ('*model_space', '*paper_space', '*paper_space0'):
                            block_def = doc.blocks[block_name]
                            new_block = new_doc.blocks.new(name=block_name)
                            for block_entity in block_def:
                                new_block.add_entity(block_entity.copy())
                            copied_block_names.add(block_name)

                            for block_entity in block_def:
                                if block_entity.dxftype() == 'INSERT':
                                    nested_block_name = block_entity.dxf.name
                                    if nested_block_name not in copied_block_names and nested_block_name in doc.blocks:
                                        nested_block = doc.blocks[nested_block_name]
                                        new_nested_block = new_doc.blocks.new(name=nested_block_name)
                                        for nested_entity in nested_block:
                                            new_nested_block.add_entity(nested_entity.copy())
                                        copied_block_names.add(nested_block_name)

                new_msp.add_entity(entity.copy())

        output_path = output_path_template.format(index + 1)

        new_doc.saveas(output_path)

def process_dxf(input_path, output_template):
    """Main processing function"""
    tablet_data = get_tablet_bounds_and_refs(input_path)

    doc = ezdxf.readfile(input_path)

    extract_entities_for_tablet(doc, tablet_data, output_template)

    return len(tablet_data)

if __name__ == "__main__":
    input_path = r"C:\srtp\dxf_split\8-040116-SOP1(ETCH) REV A.dxf"
    output_template = r"C:\srtp\dxf_split\8-040116-SOP1(ETCH) REV A_{}.dxf"

    try:
        print(f"\nStarting to process file: {input_path}")
        if not os.path.exists(input_path):
            print("Error: Input file does not exist!")
        else:
            print("File exists, starting processing...")
            num_tablets = process_dxf(input_path, output_template)
            print(f"\nProcessed {num_tablets} TABLET blocks and generated corresponding DXF files.")
    except Exception as e:
        print(f"Error during processing: {str(e)}")
        import traceback
        traceback.print_exc()