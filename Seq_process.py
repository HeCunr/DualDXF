# dataset/Seq_process.py

import os
import json
import glob
import h5py
import math
import numpy as np
import ezdxf

ENTITY_TYPES = [
    'HATCH', 'TEXT', 'MTEXT', 'LWPOLYLINE', 'ARC', 'LINE',
    'CIRCLE', 'DIMENSION', 'LEADER', 'INSERT',
    'SPLINE', 'SOLID', 'EOS'
]

FEATURE_NAMES = [
    'solid_fill', 'associative', 'boundary_paths',
    'text_insert_point_x', 'text_insert_point_y', 'height', 'text_rotation',
    'mtext_insert_point_x', 'mtext_insert_point_y', 'char_height', 'width',
    'closed', 'points_x', 'points_y', 'count',
    'arc_center_x', 'arc_center_y', 'arc_radius', 'start_angle', 'end_angle',
    'start_point_x', 'start_point_y', 'end_point_x', 'end_point_y',
    'circle_center_x', 'circle_center_y', 'circle_radius',
    'defpoint_x', 'defpoint_y', 'text_midpoint_x', 'text_midpoint_y',
    'main_value',
    'tolerance',
    'vertices_x', 'vertices_y',
    'insert_insert_point_x', 'insert_insert_point_y', 'scale_x', 'scale_y',
    'insert_rotation',
    'control_points_x', 'control_points_y', 'avg_knots',
    'solid_points_x', 'solid_points_y'
]

def parse_dxf_dimensions(file_path):
    try:
        doc = ezdxf.readfile(file_path)
    except Exception as e:
        print(f"can't read DXF {file_path}: {e}")
        return []

    msp = doc.modelspace()
    dimensions = []
    for i, entity in enumerate(msp):
        if entity.dxftype() == 'DIMENSION':
            dim_value = float(entity.dxf.measurement) if hasattr(entity.dxf, 'measurement') else 0.0
            tolerance = 0.0
            dimensions.append({
                'id': i,
                'dim_value': dim_value * 1000,
                'tolerance': tolerance * 1000,
            })
    return dimensions

def min_max_normalize(value, min_val, max_val):
    if max_val - min_val == 0:
        return 0.0
    return (value - min_val) / (max_val - min_val)

def angle_to_01(angle_deg):
    return max(0.0, min(1.0, angle_deg / 360.0))

def get_entity_points(entity):
    entity_type = entity.dxftype()
    points = []

    if entity_type == 'LINE':
        points.append(entity.dxf.start)
        points.append(entity.dxf.end)
    elif entity_type == 'LWPOLYLINE':
        points.extend([pt[:2] for pt in entity.get_points()])
    elif entity_type == 'POLYLINE':
        points.extend([v.dxf.location for v in entity.vertices])
    elif entity_type == 'CIRCLE':
        c = entity.dxf.center
        r = entity.dxf.radius
        points.append((c[0] - r, c[1]))
        points.append((c[0] + r, c[1]))
        points.append((c[0], c[1] - r))
        points.append((c[0], c[1] + r))
    elif entity_type == 'ARC':
        c = entity.dxf.center
        r = entity.dxf.radius
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle   = math.radians(entity.dxf.end_angle)
        for ang in (start_angle, end_angle):
            x = c[0] + r*math.cos(ang)
            y = c[1] + r*math.sin(ang)
            points.append((x, y))
    elif entity_type in ['TEXT', 'MTEXT']:
        ip = entity.dxf.insert
        points.append(ip)
    elif entity_type == 'HATCH':
        for path in entity.paths:
            if hasattr(path, 'edges'):
                for edge in path.edges:
                    if hasattr(edge, 'start') and hasattr(edge, 'end'):
                        points.append(edge.start)
                        points.append(edge.end)
    elif entity_type == 'DIMENSION':
        defp = entity.dxf.defpoint
        tmp  = entity.dxf.text_midpoint
        points.append(defp)
        points.append(tmp)
    elif entity_type == 'LEADER':
        points.extend(entity.vertices)
    elif entity_type == 'INSERT':
        ip = entity.dxf.insert
        points.append(ip)
    elif entity_type == 'SPLINE':
        pts = entity.control_points
        points.extend([(p[0], p[1]) for p in pts])
    elif entity_type == 'SOLID':
        v0 = (entity.dxf.vtx0.x, entity.dxf.vtx0.y)
        v1 = (entity.dxf.vtx1.x, entity.dxf.vtx1.y)
        v2 = (entity.dxf.vtx2.x, entity.dxf.vtx2.y)
        points.extend([v0, v1, v2])

    return points

def get_bounding_box(doc):
    msp = doc.modelspace()
    min_x = float('inf')
    min_y = float('inf')
    max_x = float('-inf')
    max_y = float('-inf')
    for entity in msp:
        try:
            pts = get_entity_points(entity)
            for x,y in pts:
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
        except:
            continue

    if min_x == float('inf'):
        return 0,0,0,0
    return min_x, min_y, max_x, max_y

def process_seq_file(input_dxf_path, output_dir):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        doc = ezdxf.readfile(input_dxf_path)
    except Exception as e:
        print(f"[Error] read DXF fail: {e}")
        return None

    msp = doc.modelspace()
    entity_count = len(list(msp))
    if entity_count < 50:
        print(f"skip {input_dxf_path}：number {entity_count} < 50")
        return None

    min_x, min_y, max_x, max_y = get_bounding_box(doc)
    width  = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height) if max(width, height)>0 else 1.0

    dims_info = parse_dxf_dimensions(input_dxf_path)
    dim_map = {}
    if dims_info:
        all_dim_vals = [d['dim_value'] for d in dims_info]
        all_tol_vals = [d['tolerance'] for d in dims_info]
        dim_min, dim_max = (min(all_dim_vals), max(all_dim_vals)) if max(all_dim_vals)>0 else (0,1)
        tol_min, tol_max = (min(all_tol_vals), max(all_tol_vals)) if max(all_tol_vals)>0 else (0,1)
        for d in dims_info:
            mv = min_max_normalize(d['dim_value'], dim_min, dim_max)
            tv = min_max_normalize(d['tolerance'], tol_min, tol_max)
            dim_map[d['id']] = (mv, tv)

    entities = []
    for i, entity in enumerate(msp):
        etype = entity.dxftype()
        features = {}

        if etype == 'LINE':
            sp = entity.dxf.start
            ep = entity.dxf.end
            features['start_point_x'] = min_max_normalize(sp[0], min_x, max_x)
            features['start_point_y'] = min_max_normalize(sp[1], min_y, max_y)
            features['end_point_x']   = min_max_normalize(ep[0], min_x, max_x)
            features['end_point_y']   = min_max_normalize(ep[1], min_y, max_y)

        elif etype == 'CIRCLE':
            c = entity.dxf.center
            r = entity.dxf.radius
            features['circle_center_x'] = min_max_normalize(c[0], min_x, max_x)
            features['circle_center_y'] = min_max_normalize(c[1], min_y, max_y)
            features['circle_radius']   = r/max_dim if max_dim>0 else 0.0

        elif etype == 'ARC':
            c = entity.dxf.center
            r = entity.dxf.radius
            sa = entity.dxf.start_angle
            ea = entity.dxf.end_angle
            features['arc_center_x'] = min_max_normalize(c[0], min_x, max_x)
            features['arc_center_y'] = min_max_normalize(c[1], min_y, max_y)
            features['arc_radius']   = r/max_dim
            features['start_angle']  = angle_to_01(sa)
            features['end_angle']    = angle_to_01(ea)

        elif etype == 'LWPOLYLINE':
            closed = int(entity.closed)
            pts = [p[:2] for p in entity.get_points()]
            features['closed'] = closed
            features['count']  = len(pts)
            if pts:
                avg_x = sum(p[0] for p in pts)/len(pts)
                avg_y = sum(p[1] for p in pts)/len(pts)
                features['points_x'] = min_max_normalize(avg_x, min_x, max_x)
                features['points_y'] = min_max_normalize(avg_y, min_y, max_y)

        elif etype == 'TEXT':
            ip = entity.dxf.insert
            h  = entity.dxf.height
            rot= entity.dxf.rotation
            features['text_insert_point_x'] = min_max_normalize(ip[0], min_x, max_x)
            features['text_insert_point_y'] = min_max_normalize(ip[1], min_y, max_y)
            features['height']              = h/max_dim
            features['text_rotation']       = angle_to_01(rot)

        elif etype == 'MTEXT':
            ip = entity.dxf.insert
            ch = entity.dxf.char_height
            wd = entity.dxf.width
            rot= entity.dxf.rotation
            features['mtext_insert_point_x'] = min_max_normalize(ip[0], min_x, max_x)
            features['mtext_insert_point_y'] = min_max_normalize(ip[1], min_y, max_y)
            features['char_height']          = ch/max_dim
            features['width']                = wd/max_dim
            features['text_rotation']        = angle_to_01(rot)

        elif etype == 'HATCH':
            sf = int(entity.dxf.solid_fill) if hasattr(entity.dxf, 'solid_fill') else 0
            asso = int(entity.dxf.associative) if hasattr(entity.dxf, 'associative') else 0
            paths_count = len(entity.paths)
            features['solid_fill']    = sf
            features['associative']   = asso
            features['boundary_paths']= paths_count

        elif etype == 'DIMENSION':
            defp = entity.dxf.defpoint
            tmid= entity.dxf.text_midpoint
            dim_id = i
            features['defpoint_x']       = min_max_normalize(defp[0], min_x, max_x)
            features['defpoint_y']       = min_max_normalize(defp[1], min_y, max_y)
            features['text_midpoint_x']  = min_max_normalize(tmid[0], min_x, max_x)
            features['text_midpoint_y']  = min_max_normalize(tmid[1], min_y, max_y)
            if dim_id in dim_map:
                mv, tv = dim_map[dim_id]
                features['main_value'] = mv
                features['tolerance']  = tv
            else:
                features['main_value'] = 0.0
                features['tolerance']  = 0.0

        elif etype == 'LEADER':
            verts = entity.vertices
            if verts:
                avg_x = sum(v[0] for v in verts)/len(verts)
                avg_y = sum(v[1] for v in verts)/len(verts)
                features['vertices_x'] = min_max_normalize(avg_x, min_x, max_x)
                features['vertices_y'] = min_max_normalize(avg_y, min_y, max_y)
            anno_type = getattr(entity.dxf, 'annotation_type', 0)
            features['annotation_type'] = anno_type

        elif etype == 'INSERT':
            ip = entity.dxf.insert
            sx = entity.dxf.xscale
            sy = entity.dxf.yscale
            rot= entity.dxf.rotation
            features['insert_insert_point_x'] = min_max_normalize(ip[0], min_x, max_x)
            features['insert_insert_point_y'] = min_max_normalize(ip[1], min_y, max_y)
            features['scale_x'] = sx
            features['scale_y'] = sy
            features['insert_rotation'] = angle_to_01(rot)

        elif etype == 'SPLINE':
            cpts = entity.control_points
            if cpts:
                avg_x = sum(p[0] for p in cpts)/len(cpts)
                avg_y = sum(p[1] for p in cpts)/len(cpts)
                features['control_points_x'] = min_max_normalize(avg_x, min_x, max_x)
                features['control_points_y'] = min_max_normalize(avg_y, min_y, max_y)
            kts = entity.knots
            if kts:
                avg_k = sum(kts)/len(kts)
                features['avg_knots'] = avg_k
            else:
                features['avg_knots'] = 0.0

        elif etype == 'SOLID':
            v0 = (entity.dxf.vtx0.x, entity.dxf.vtx0.y)
            v1 = (entity.dxf.vtx1.x, entity.dxf.vtx1.y)
            v2 = (entity.dxf.vtx2.x, entity.dxf.vtx2.y)
            arr = [v0,v1,v2]
            avg_x = sum(a[0] for a in arr)/3.0
            avg_y = sum(a[1] for a in arr)/3.0
            features['solid_points_x'] = min_max_normalize(avg_x, min_x, max_x)
            features['solid_points_y'] = min_max_normalize(avg_y, min_y, max_y)

        else:
            continue

        entity_dict = {
            'type': etype,
            'features': features
        }
        entities.append(entity_dict)

    basename = os.path.splitext(os.path.basename(input_dxf_path))[0]
    out_json = os.path.join(output_dir, basename + '.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(entities, f, ensure_ascii=False, indent=2)
    return out_json

def json_to_dxf_vector(json_file, max_len=4096):

    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    vectors = []
    for ent in data:
        etype = ent['type']
        feat  = ent['features']
        row   = [-1.0]*(len(FEATURE_NAMES)+1)
        if etype in ENTITY_TYPES:
            row[0] = float(ENTITY_TYPES.index(etype))
        else:
            row[0] = -1.0

        for i, fn in enumerate(FEATURE_NAMES):
            val = feat.get(fn, None)
            if val is None:
                continue
            if isinstance(val, bool):
                row[i+1] = float(val)
            else:
                row[i+1] = float(val)

        vectors.append(row)

    if len(vectors) > max_len:
        return None

    eos_row = [-1.0]*(len(FEATURE_NAMES)+1)
    eos_row[0] = float(ENTITY_TYPES.index('EOS'))
    while len(vectors) < max_len:
        vectors.append(eos_row)

    arr = np.array(vectors, dtype=np.float32)
    return arr

def process_seq_files_batch(input_dir, output_h5_dir):

    if not os.path.exists(output_h5_dir):
        os.makedirs(output_h5_dir)

    dxf_files = glob.glob(os.path.join(input_dir, '*.dxf'))
    if not dxf_files:
        print(f" {input_dir} can't find dxf。")
        return

    temp_json_dir = os.path.join(output_h5_dir, 'temp_json')
    os.makedirs(temp_json_dir, exist_ok=True)

    for dxf_file in dxf_files:
        print(f"process: {dxf_file}")
        out_json = process_seq_file(dxf_file, temp_json_dir)
        if out_json is None:
            continue

        dxf_vec = json_to_dxf_vector(out_json, max_len=4096)
        if dxf_vec is None:
            print(f"  -> {dxf_file} skip.")
            continue

        base = os.path.splitext(os.path.basename(dxf_file))[0]
        h5_path = os.path.join(output_h5_dir, base + '.h5')
        with h5py.File(h5_path, 'w') as hf:
            hf.create_dataset('dxf_vec', data=dxf_vec[np.newaxis, ...], dtype=np.float32)

        print(f"  -> done: {h5_path}")

    print("all done")

if __name__ == '__main__':
    input_dir =  r"/home/vllm/DualDXF/data/DXF/SuperLFD_train"
    output_h5_dir =  r"/home/vllm/DualDXF/data/Seq/SuperLFD_train"

    process_seq_files_batch(input_dir, output_h5_dir)
