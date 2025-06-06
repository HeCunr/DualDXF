import json
import os

def process_entity(entity):
    entity_type = entity['type']
    features = entity['features']
    new_features = {}

    if entity_type == 'HATCH':
        new_features['solid_fill'] = features.get('solid_fill', 0)
        new_features['associative'] = features.get('associative', 0)
        new_features['boundary_paths'] = features.get('boundary_paths', 0)

    elif entity_type == 'TEXT':
        point = features.get('text_insert_point', [0, 0])
        new_features['text_insert_point_x'] = point[0]
        new_features['text_insert_point_y'] = point[1]
        new_features['height'] = features.get('height', 0)
        new_features['text_rotation'] = features.get('rotation', 0)

    elif entity_type == 'MTEXT':
        point = features.get('mtext_insert_point', [0, 0])
        new_features['mtext_insert_point_x'] = point[0]
        new_features['mtext_insert_point_y'] = point[1]
        new_features['char_height'] = features.get('char_height', 0)
        new_features['width'] = features.get('width', 0)

    elif entity_type == 'LWPOLYLINE':
        points = features.get('points', [])
        if points:
            x_sum = sum(p[0] for p in points)
            y_sum = sum(p[1] for p in points)
            avg_x = x_sum / len(points)
            avg_y = y_sum / len(points)
        else:
            avg_x = avg_y = 0
        new_features['closed'] = features.get('closed', False)
        new_features['points_x'] = avg_x
        new_features['points_y'] = avg_y
        new_features['count'] = features.get('count', 0)

    elif entity_type == 'ARC':
        point = features.get('arc_center', [0, 0])
        new_features['arc_center_x'] = point[0]
        new_features['arc_center_y'] = point[1]
        new_features['arc_radius'] = features.get('radius', 0)
        new_features['start_angle'] = features.get('start_angle', 0)
        new_features['end_angle'] = features.get('end_angle', 0)

    elif entity_type == 'LINE':
        start_point = features.get('start_point', [0, 0])
        end_point = features.get('end_point', [0, 0])
        new_features['start_point_x'] = start_point[0]
        new_features['start_point_y'] = start_point[1]
        new_features['end_point_x'] = end_point[0]
        new_features['end_point_y'] = end_point[1]

    elif entity_type == 'CIRCLE':
        point = features.get('circle_center', [0, 0])
        new_features['circle_center_x'] = point[0]
        new_features['circle_center_y'] = point[1]
        new_features['circle_radius'] = features.get('circle_radius', 0)

    elif entity_type == 'DIMENSION':
        defpoint = features.get('defpoint', [0, 0])
        text_midpoint = features.get('text_midpoint', [0, 0])
        new_features['defpoint_x'] = defpoint[0]
        new_features['defpoint_y'] = defpoint[1]
        new_features['text_midpoint_x'] = text_midpoint[0]
        new_features['text_midpoint_y'] = text_midpoint[1]
        new_features['dim_type'] = features.get('dim_type', 0)

    elif entity_type == 'LEADER':
        vertices = features.get('vertices', [])
        if vertices:
            x_sum = sum(v[0] for v in vertices)
            y_sum = sum(v[1] for v in vertices)
            avg_x = x_sum / len(vertices)
            avg_y = y_sum / len(vertices)
        else:
            avg_x = avg_y = 0
        new_features['vertices_x'] = avg_x
        new_features['vertices_y'] = avg_y
        new_features['annotation_type'] = features.get('annotation_type', 0)

    elif entity_type == 'INSERT':
        point = features.get('insert_insert_point', [0, 0])
        new_features['insert_insert_point_x'] = point[0]
        new_features['insert_insert_point_y'] = point[1]
        scale = features.get('scale', (1, 1))
        new_features['scale_x'] = scale[0]
        new_features['scale_y'] = scale[1]
        new_features['insert_rotation'] = features.get('insert_rotation', 0)

    else:
        new_features = features

    return {
        'type': entity_type,
        'features': new_features
    }

def process_json_file(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    processed_entities = []
    for entity in data:
        processed_entity = process_entity(entity)
        processed_entities.append(processed_entity)

    with open(output_path, 'w') as f:
        json.dump(processed_entities, f, indent=2)

    print(f" {input_path}，save {output_path}")

def main():
    input_dir = r'C:\srtp\encode\data\raw'

    output_dir = r'C:\srtp\encode\data\new'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith('.json'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            process_json_file(input_path, output_path)

if __name__ == '__main__':
    main()
