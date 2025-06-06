import ezdxf
import re
def extract_insert_features(entity):
    return {
        'name': entity.dxf.name,
        'insert_point': entity.dxf.insert,
        'scale': (entity.dxf.xscale, entity.dxf.yscale, entity.dxf.zscale),
        'rotation': entity.dxf.rotation
    }

def extract_line_features(entity):
    return {
        'start_point': entity.dxf.start,
        'end_point': entity.dxf.end
    }

def extract_text_features(entity):
    return {
        'text': entity.dxf.text,
        'insert_point': entity.dxf.insert,
        'height': entity.dxf.height,
        'rotation': entity.dxf.rotation
    }

def extract_mtext_features(entity):
    return {
        'text': entity.text,
        'insert_point': entity.dxf.insert,
        'char_height': entity.dxf.char_height,
        'width': entity.dxf.width,
        'rotation': entity.dxf.rotation
    }

def extract_hatch_features(entity):
    return {
        'pattern_name': entity.dxf.pattern_name,
        'solid_fill': entity.dxf.solid_fill,
        'associative': entity.dxf.associative,
        'boundary_paths': len(entity.paths)
    }

def extract_lwpolyline_features(entity):
    return {
        'closed': entity.closed,
        'points': list(entity.get_points()),
        'count': len(entity)
    }

def extract_leader_features(entity):
    return {
        'vertices': list(entity.vertices),
        'annotation_type': entity.dxf.annotation_type
    }

def extract_circle_features(entity):
    return {
        'center': entity.dxf.center,
        'radius': entity.dxf.radius
    }

def parse_text(text):
    # Match format: "0.203%%P0.008"
    match1 = re.search(r'(\d+\.\d+)?%%[Pp]?(\d+\.\d+)?', text)
    if match1:
        text_0 = match1.group(1) if match1.group(1) else "0"
        text_1 = match1.group(2) if match1.group(2) else "0"
        return float(text_0), float(text_1)

    # Match format: "0.100±0.030"
    match2 = re.search(r'(\d+\.\d+)±(\d+\.\d+)', text)
    if match2:
        text_0 = match2.group(1)
        text_1 = match2.group(2)
        return float(text_0), float(text_1)

    # Match format: just a number
    match3 = re.match(r'^(\d+\.\d+)$', text)
    if match3:
        text_0 = match3.group(1)
        text_1 = "0"
        return float(text_0), int(text_1)

    return 0, 0

def extract_line_dim(dim, doc):
    block = doc.blocks.get(dim.dxf.geometry)
    if block is not None:
        for entity in block:
            if entity.dxftype() in ['TEXT', 'MTEXT']:
                text = entity.dxf.text
                value_0 = extract_numbers(text)
                return value_0

def extract_numbers(text):
    pattern = r'\\A1;(\d+\.\d+)'
    match = re.search(pattern, text)
    if match:
        number = match.group(1) if match.group(1) else "0"
        return float(number)
    else:
        return 0

def extract_dimension_features(entity, doc):
    dim_text = entity.dxf.text
    value_0, value_1 = parse_text(dim_text)

    if value_0 == 0:
        value_0 = extract_line_dim(entity, doc)

    return {
        'defpoint': entity.dxf.defpoint,
        'text_midpoint': entity.dxf.text_midpoint,
        'dim_type': entity.dimtype,
        'text': dim_text,
        'value': value_0,
        'tolerance': value_1
    }

def extract_arc_features(entity):
    return {
        'center': entity.dxf.center,
        'radius': entity.dxf.radius,
        'start_angle': entity.dxf.start_angle,
        'end_angle': entity.dxf.end_angle
    }

def extract_solid_features(entity):
    return {
        'points': [
            (entity.dxf.vtx0.x, entity.dxf.vtx0.y),
            (entity.dxf.vtx1.x, entity.dxf.vtx1.y),
            (entity.dxf.vtx2.x, entity.dxf.vtx2.y)
        ]
    }

def extract_spline_features(entity):
    return {
        'degree': entity.dxf.degree,
        'control_points': [(point[0], point[1]) for point in entity.control_points],
        'knots': list(entity.knots)
    }

def extract_entity_features(entity):
    entity_type = entity.dxftype()
    if entity_type == 'INSERT':
        return extract_insert_features(entity)
    elif entity_type == 'LINE':
        return extract_line_features(entity)
    elif entity_type == 'TEXT':
        return extract_text_features(entity)
    elif entity_type == 'MTEXT':
        return extract_mtext_features(entity)
    elif entity_type == 'HATCH':
        return extract_hatch_features(entity)
    elif entity_type == 'LWPOLYLINE':
        return extract_lwpolyline_features(entity)
    elif entity_type == 'LEADER':
        return extract_leader_features(entity)
    elif entity_type == 'CIRCLE':
        return extract_circle_features(entity)
    elif entity_type == 'DIMENSION':
        return extract_dimension_features(entity)
    elif entity_type == 'ARC':
        return extract_arc_features(entity)
    elif entity_type == 'SOLID':
        return extract_solid_features(entity)
    elif entity_type == 'SPLINE':
        return extract_spline_features(entity)
    else:
        return {'error': f'Unsupported entity type: {entity_type}'}

def process_dxf_file(file_path):
    doc = ezdxf.readfile(file_path)
    msp = doc.modelspace()

    entities = {}
    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type not in entities:
            entities[entity_type] = []
        if entity_type == 'DIMENSION':
            entities[entity_type].append(extract_dimension_features(entity, doc))
        else:
            entities[entity_type].append(extract_entity_features(entity))

    return entities

if __name__ == "__main__":
    file_path =  r'C:\srtp\241101\TRAIN\10L185150DCE Rev0_3.dxf'

    extracted_features = process_dxf_file(file_path)

    for entity_type, features in extracted_features.items():
       if entity_type == 'SOLID':
           print(f'Found {len(features)} {entity_type} entities')
           for feature in features:
               print(feature)