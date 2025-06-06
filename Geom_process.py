"""
dataset/Geom_process.py

"""
import os
import json
import math
import ezdxf
import numpy as np

from dxflib.GeomLib.box import compute_entity_bounding_box
from Seq_process import (
    get_bounding_box,
)

from dxflib.DxfLib.parse_text import (
    extract_dimension_values,
)

ENTITY_TYPE_ORDER = [
    'HATCH', 'TEXT', 'MTEXT', 'LWPOLYLINE', 'ARC',
    'LINE', 'CIRCLE', 'DIMENSION', 'LEADER', 'INSERT',
    'SPLINE', 'SOLID'
]
ENTITY_TYPE_MAP = {etype: idx for idx, etype in enumerate(ENTITY_TYPE_ORDER)}

#  0:  entity_type
#  1:  solid_fill
#  2:  associative
#  3:  boundary_paths
#  4:  text_insert_point_x
#  5:  text_insert_point_y
#  6:  height
#  7:  text_rotation
#  8:  mtext_insert_point_x
#  9:  mtext_insert_point_y
# 10:  char_height
# 11:  width
# 12:  closed
# 13:  points_x
# 14:  points_y
# 15:  count
# 16:  arc_center_x
# 17:  arc_center_y
# 18:  arc_radius
# 19:  start_angle
# 20:  end_angle
# 21:  start_point_x
# 22:  start_point_y
# 23:  end_point_x
# 24:  end_point_y
# 25:  circle_center_x
# 26:  circle_center_y
# 27:  circle_radius
# 28:  defpoint_x
# 29:  defpoint_y
# 30:  text_midpoint_x
# 31:  text_midpoint_y
# 32:  main_value      (Dimension)
# 33:  tolerance       (Dimension)
# 34:  vertices_x      (Leader)
# 35:  vertices_y      (Leader)
# 36:  insert_insert_point_x
# 37:  insert_insert_point_y
# 38:  scale_x
# 39:  scale_y
# 40:  insert_rotation
# 41:  control_points_x (Spline)
# 42:  control_points_y (Spline)
# 43:  avg_knots        (Spline)
# 44:  solid_points_x   (Solid)
# 45:  solid_points_y   (Solid)

def clamp_to_0_1(value):
    if value < 0.0:
        return 0.0
    elif value > 1.0:
        return 1.0
    else:
        return value

def norm_coord(x, min_x, max_x):

    if max_x > min_x:
        return clamp_to_0_1((x - min_x) / (max_x - min_x))
    else:
        return 0.0

def norm_length(v, max_dim):

    if max_dim > 0:
        return clamp_to_0_1(v / max_dim)
    else:
        return 0.0

def norm_angle(a):

    return clamp_to_0_1(a / 360.0)

def box_contained_in(box1, box2) -> bool:
    (min1x, min1y, max1x, max1y) = box1
    (min2x, min2y, max2x, max2y) = box2
    return (min1x > min2x and min1y > min2y and
            max1x < max2x and max1y < max2y)

def boxes_adjacent(box1, box2) -> bool:
    if box_contained_in(box1, box2) or box_contained_in(box2, box1):
        return False

    min1x, min1y, max1x, max1y = box1
    min2x, min2y, max2x, max2y = box2
    if max1x < min2x or min1x > max2x:
        return False
    if max1y < min2y or min1y > max2y:
        return False
    return True

def gather_and_normalize_dimensions(dimension_entities, doc):

    if not dimension_entities:
        return {}

    all_main_values = []
    all_tolerances = []
    dim_map = {}  # handle => (raw_main_val, raw_tol)

    for ent in dimension_entities:
        main_val, tol_val = extract_dimension_values(ent, doc)  # mm*1000
        ent_handle = ent.dxf.handle

        dim_map[ent_handle] = (main_val, tol_val)
        all_main_values.append(main_val)
        all_tolerances.append(tol_val)

    main_min = min(all_main_values) if any(v > 0 for v in all_main_values) else 0.0
    main_max = max(all_main_values) if any(v > 0 for v in all_main_values) else 1.0
    tol_min  = min(all_tolerances) if any(v > 0 for v in all_tolerances) else 0.0
    tol_max  = max(all_tolerances) if any(v > 0 for v in all_tolerances) else 1.0

    norm_map = {}
    def safe_norm(val, vmin, vmax):
        if vmax > vmin:
            return clamp_to_0_1((val - vmin)/(vmax - vmin))
        else:
            return 0.0

    for handle, (m_raw, t_raw) in dim_map.items():
        m_norm = safe_norm(m_raw, main_min, main_max)
        t_norm = safe_norm(t_raw, tol_min, tol_max)
        norm_map[handle] = (m_norm, t_norm)

    return norm_map

def extract_features(entity, doc,
                     min_x, min_y, max_x, max_y, max_dim,
                     dim_norm_map=None) -> list:

    row = [0]*46

    etype = entity.dxftype()
    if etype not in ENTITY_TYPE_MAP:
        return row

    row[0] = ENTITY_TYPE_MAP[etype]

    try:
        #-------------- HATCH --------------
        if etype == 'HATCH':
            if hasattr(entity.dxf, 'solid_fill'):
                row[1] = float(entity.dxf.solid_fill)
            if hasattr(entity.dxf, 'associative'):
                row[2] = float(entity.dxf.associative)
            if hasattr(entity, 'paths'):
                boundary_paths = len(entity.paths)
                row[3] = float(boundary_paths)

        #-------------- TEXT ---------------
        elif etype == 'TEXT':
            if hasattr(entity.dxf, 'insert'):
                ins = entity.dxf.insert
                row[4] = norm_coord(ins[0], min_x, max_x)
                row[5] = norm_coord(ins[1], min_y, max_y)
            if hasattr(entity.dxf, 'height'):
                row[6] = norm_length(entity.dxf.height, max_dim)
            if hasattr(entity.dxf, 'rotation'):
                row[7] = norm_angle(entity.dxf.rotation)

        #-------------- MTEXT --------------
        elif etype == 'MTEXT':
            if hasattr(entity.dxf, 'insert'):
                ins = entity.dxf.insert
                row[8]  = norm_coord(ins[0], min_x, max_x)
                row[9]  = norm_coord(ins[1], min_y, max_y)
            if hasattr(entity.dxf, 'char_height'):
                row[10] = norm_length(entity.dxf.char_height, max_dim)
            if hasattr(entity.dxf, 'width'):
                row[11] = norm_length(entity.dxf.width, max_dim)

        #----------- LWPOLYLINE -----------
        elif etype == 'LWPOLYLINE':
            if hasattr(entity, 'closed'):
                row[12] = float(entity.closed)
            if hasattr(entity, 'get_points'):
                points = list(entity.get_points())
                count_ = len(points)
                row[15] = float(count_)
                if count_ > 0:
                    x_sum = sum(p[0] for p in points)
                    y_sum = sum(p[1] for p in points)
                    avg_x = x_sum / count_
                    avg_y = y_sum / count_
                    row[13] = norm_coord(avg_x, min_x, max_x)
                    row[14] = norm_coord(avg_y, min_y, max_y)

        #--------------- ARC ---------------
        elif etype == 'ARC':
            if hasattr(entity.dxf, 'center'):
                cx, cy, _ = entity.dxf.center
                row[16] = norm_coord(cx, min_x, max_x)
                row[17] = norm_coord(cy, min_y, max_y)
            if hasattr(entity.dxf, 'radius'):
                row[18] = norm_length(entity.dxf.radius, max_dim)
            if hasattr(entity.dxf, 'start_angle'):
                row[19] = norm_angle(entity.dxf.start_angle)
            if hasattr(entity.dxf, 'end_angle'):
                row[20] = norm_angle(entity.dxf.end_angle)

        #-------------- LINE ---------------
        elif etype == 'LINE':
            if hasattr(entity.dxf, 'start'):
                sp = entity.dxf.start
                row[21] = norm_coord(sp[0], min_x, max_x)
                row[22] = norm_coord(sp[1], min_y, max_y)
            if hasattr(entity.dxf, 'end'):
                ep = entity.dxf.end
                row[23] = norm_coord(ep[0], min_x, max_x)
                row[24] = norm_coord(ep[1], min_y, max_y)

        #------------- CIRCLE --------------
        elif etype == 'CIRCLE':
            if hasattr(entity.dxf, 'center'):
                cx, cy, _ = entity.dxf.center
                row[25] = norm_coord(cx, min_x, max_x)
                row[26] = norm_coord(cy, min_y, max_y)
            if hasattr(entity.dxf, 'radius'):
                row[27] = norm_length(entity.dxf.radius, max_dim)

        #----------- DIMENSION ------------
        elif etype == 'DIMENSION':
            if hasattr(entity.dxf, 'defpoint'):
                dx, dy, _ = entity.dxf.defpoint
                row[28] = norm_coord(dx, min_x, max_x)
                row[29] = norm_coord(dy, min_y, max_y)
            if hasattr(entity.dxf, 'text_midpoint'):
                mx, my, _ = entity.dxf.text_midpoint
                row[30] = norm_coord(mx, min_x, max_x)
                row[31] = norm_coord(my, min_y, max_y)

            if dim_norm_map is not None:
                handle = entity.dxf.handle
                if handle in dim_norm_map:
                    main_v, tol_v = dim_norm_map[handle]  # [0,1]
                    row[32] = main_v
                    row[33] = tol_v

        #-------------- LEADER -------------
        elif etype == 'LEADER':
            if hasattr(entity, 'vertices'):
                vertices = entity.vertices
                if len(vertices) > 0:
                    x_sum = sum(v[0] for v in vertices)
                    y_sum = sum(v[1] for v in vertices)
                    avg_x = x_sum / len(vertices)
                    avg_y = y_sum / len(vertices)
                    row[34] = norm_coord(avg_x, min_x, max_x)
                    row[35] = norm_coord(avg_y, min_y, max_y)

        #------------- INSERT --------------
        elif etype == 'INSERT':
            if hasattr(entity.dxf, 'insert'):
                ins = entity.dxf.insert
                row[36] = norm_coord(ins[0], min_x, max_x)
                row[37] = norm_coord(ins[1], min_y, max_y)
            if hasattr(entity.dxf, 'xscale'):
                row[38] = float(entity.dxf.xscale)
            if hasattr(entity.dxf, 'yscale'):
                row[39] = float(entity.dxf.yscale)
            if hasattr(entity.dxf, 'rotation'):
                row[40] = norm_angle(entity.dxf.rotation)

        #------------- SPLINE --------------
        elif etype == 'SPLINE':
            if hasattr(entity, 'control_points') and entity.control_points:
                cpoints = entity.control_points
                x_sum = sum(p[0] for p in cpoints)
                y_sum = sum(p[1] for p in cpoints)
                avg_x = x_sum / len(cpoints)
                avg_y = y_sum / len(cpoints)
                row[41] = norm_coord(avg_x, min_x, max_x)
                row[42] = norm_coord(avg_y, min_y, max_y)
            if hasattr(entity, 'knots') and entity.knots:
                knots = entity.knots
                avg_k = sum(knots) / len(knots)
                row[43] = clamp_to_0_1(avg_k)

        #-------------- SOLID --------------
        elif etype == 'SOLID':
            vtxs = []
            if hasattr(entity.dxf, 'vtx0'):
                vtxs.append((entity.dxf.vtx0.x, entity.dxf.vtx0.y))
            if hasattr(entity.dxf, 'vtx1'):
                vtxs.append((entity.dxf.vtx1.x, entity.dxf.vtx1.y))
            if hasattr(entity.dxf, 'vtx2'):
                vtxs.append((entity.dxf.vtx2.x, entity.dxf.vtx2.y))
            if hasattr(entity.dxf, 'vtx3'):
                vtxs.append((entity.dxf.vtx3.x, entity.dxf.vtx3.y))
            vtxs = [v for v in vtxs if v is not None]
            if len(vtxs) > 0:
                x_sum = sum(p[0] for p in vtxs)
                y_sum = sum(p[1] for p in vtxs)
                avg_x = x_sum / len(vtxs)
                avg_y = y_sum / len(vtxs)
                row[44] = norm_coord(avg_x, min_x, max_x)
                row[45] = norm_coord(avg_y, min_y, max_y)

    except Exception:
        pass

    return row

def process_single_dxf(dxf_path: str, output_dir: str = None):

    if not os.path.isfile(dxf_path):
        print(f"don't exist：{dxf_path}")
        return

    try:
        doc = ezdxf.readfile(dxf_path)
    except Exception as e:
        print(f"can't read: {dxf_path},{e}")
        return

    msp = doc.modelspace()
    valid_entities = []
    for e in msp:
        if e.dxftype() in ENTITY_TYPE_MAP:
            valid_entities.append(e)

    num_entities = len(valid_entities)
    if num_entities < 512:
        print(f"{os.path.basename(dxf_path)} _({num_entities}) < 512")
        return
    if num_entities > 4096:
        print(f"{os.path.basename(dxf_path)} _({num_entities}) > 4096")
        return

    valid_entities.sort(key=lambda ent: int(ent.dxf.handle, 16))

    min_x, min_y, max_x, max_y = get_bounding_box(doc)
    width = max_x - min_x
    height = max_y - min_y
    max_dim = max(width, height) if max(width, height) > 1e-9 else 1.0

    dimension_entities = [ent for ent in valid_entities if ent.dxftype() == 'DIMENSION']
    dim_norm_map = gather_and_normalize_dimensions(dimension_entities, doc)

    n = len(valid_entities)
    features = np.zeros((n, 46), dtype=np.float32)

    for i, ent in enumerate(valid_entities):
        row = extract_features(
            ent, doc,
            min_x, min_y, max_x, max_y, max_dim,
            dim_norm_map=dim_norm_map
        )
        features[i, :] = np.array(row, dtype=np.float32)

    bounding_boxes = []
    for ent in valid_entities:
        box = compute_entity_bounding_box(ent, doc)
        if box is None:
            bounding_boxes.append((0, 0, 0, 0))
        else:
            bounding_boxes.append(box)

    succs = [[] for _ in range(n)]
    for i in range(n):
        box_i = bounding_boxes[i]
        for j in range(i+1, n):
            box_j = bounding_boxes[j]
            if boxes_adjacent(box_i, box_j):
                succs[i].append(j)
                succs[j].append(i)

    two_d_index = []
    for (bx_min, by_min, bx_max, by_max) in bounding_boxes:
        cx = (bx_min + bx_max) * 0.5
        cy = (by_min + by_max) * 0.5
        nx = norm_coord(cx, min_x, max_x)
        ny = norm_coord(cy, min_y, max_y)
        two_d_index.append([nx, ny])

    dxf_name = os.path.basename(dxf_path)
    result_dict = {
        "src": dxf_name,
        "n_num": n,
        "succs": succs,
        "features": features.tolist(),
        "2D-index": two_d_index
    }

    if not output_dir:
        output_dir = os.path.dirname(dxf_path)
    base_name = os.path.splitext(dxf_name)[0]
    out_json_path = os.path.join(output_dir, f"{base_name}.json")

    with open(out_json_path, "w", encoding="utf-8") as f:
        line = json.dumps(result_dict, ensure_ascii=False)
        f.write(line + "\n")

    print(f"{dxf_path} => {out_json_path}")

if __name__ == "__main__":
    import glob

    input_dir = r"/home/vllm/DualDXF/data/DXF/SuperLFD_train"
    output_dir = r"/home/vllm/DualDXF/data/Geom/SuperLFD_train"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dxf_files = glob.glob(os.path.join(input_dir, "*.dxf"))
    for dxf_file in dxf_files:
        process_single_dxf(dxf_file, output_dir)
