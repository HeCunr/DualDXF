# utils/Seq_augment.py

import os
import h5py
import numpy as np

def random_shift_float(value, shift_range=0.05):

    if value < 0.0:
        return value
    delta = np.random.uniform(-shift_range, shift_range)
    new_val = value + delta
    return np.clip(new_val, 0.0, 1.0)

def random_scale_float(value, scale_min=0.8, scale_max=1.2):

    if value < 0.0:
        return value
    factor = np.random.uniform(scale_min, scale_max)
    new_val = value * factor
    return np.clip(new_val, 0.0, 1.0)

def random_toggle_bool(value, prob=0.2):

    if np.isclose(value, 0.0) and np.random.rand() < prob:
        return 1.0
    if np.isclose(value, 1.0) and np.random.rand() < prob:
        return 0.0
    return value

def random_angle_shift(value, shift_range=0.1):

    if value < 0.0:
        return value
    delta = np.random.uniform(-shift_range, shift_range)
    new_val = value + delta
    return np.clip(new_val, 0.0, 1.0)


ENTITY_TYPES = [
    'HATCH', 'TEXT', 'MTEXT', 'LWPOLYLINE', 'ARC', 'LINE',
    'CIRCLE', 'DIMENSION', 'LEADER', 'INSERT', 'SPLINE', 'SOLID', 'EOS'
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

feature_index_map = {name: i for i, name in enumerate(FEATURE_NAMES)}


def augment_entity_line(entity_row):

    entity_type_idx = int(entity_row[0])
    if entity_type_idx < 0 or entity_type_idx >= len(ENTITY_TYPES):
        return entity_row
    if entity_type_idx == ENTITY_TYPES.index('EOS'):
        return entity_row

    entity_type = ENTITY_TYPES[entity_type_idx]

    # LINE
    if entity_type == 'LINE':
        for feat_name in ['start_point_x', 'start_point_y', 'end_point_x', 'end_point_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)

    # CIRCLE
    elif entity_type == 'CIRCLE':
        for feat_name in ['circle_center_x', 'circle_center_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
        col_r = feature_index_map['circle_radius'] + 1

        if entity_row[col_r] >= 0.0:
            if np.random.rand() < 0.5:
                entity_row[col_r] = random_shift_float(entity_row[col_r], shift_range=0.05)
            else:
                entity_row[col_r] = random_scale_float(entity_row[col_r], 0.8, 1.2)

    # ARC
    elif entity_type == 'ARC':
        for feat_name in ['arc_center_x', 'arc_center_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
        col_radius = feature_index_map['arc_radius'] + 1
        if entity_row[col_radius] >= 0.0:
            if np.random.rand() < 0.5:
                entity_row[col_radius] = random_shift_float(entity_row[col_radius], shift_range=0.05)
            else:
                entity_row[col_radius] = random_scale_float(entity_row[col_radius], 0.8, 1.2)
        for feat_name in ['start_angle', 'end_angle']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_angle_shift(entity_row[col], shift_range=0.1)

    # LWPOLYLINE
    elif entity_type == 'LWPOLYLINE':
        col_closed = feature_index_map['closed'] + 1
        entity_row[col_closed] = random_toggle_bool(entity_row[col_closed], prob=0.2)
        # points_x / points_y => shift
        for feat_name in ['points_x', 'points_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)

    # TEXT
    elif entity_type == 'TEXT':
        for feat_name in ['text_insert_point_x', 'text_insert_point_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
        # height => shift or scale
        col_h = feature_index_map['height'] + 1
        if entity_row[col_h] >= 0.0:
            if np.random.rand() < 0.5:
                entity_row[col_h] = random_shift_float(entity_row[col_h], shift_range=0.05)
            else:
                entity_row[col_h] = random_scale_float(entity_row[col_h], 0.8, 1.2)
        # text_rotation => angle
        col_rot = feature_index_map['text_rotation'] + 1
        entity_row[col_rot] = random_angle_shift(entity_row[col_rot], shift_range=0.1)

    # MTEXT
    elif entity_type == 'MTEXT':
        for feat_name in ['mtext_insert_point_x', 'mtext_insert_point_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
        for feat_name in ['char_height', 'width']:
            col = feature_index_map[feat_name] + 1
            if entity_row[col] >= 0.0:
                if np.random.rand() < 0.5:
                    entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
                else:
                    entity_row[col] = random_scale_float(entity_row[col], 0.8, 1.2)
        col_rot = feature_index_map['text_rotation'] + 1
        entity_row[col_rot] = random_angle_shift(entity_row[col_rot], shift_range=0.1)

    # HATCH
    elif entity_type == 'HATCH':
        for feat_name in ['solid_fill', 'associative']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_toggle_bool(entity_row[col], prob=0.2)
        col_bp = feature_index_map['boundary_paths'] + 1
        if entity_row[col_bp] >= 0.0:
            delta = np.random.randint(-2, 3)
            new_val = entity_row[col_bp] + delta
            entity_row[col_bp] = max(0.0, new_val)

    # DIMENSION
    elif entity_type == 'DIMENSION':
        for feat_name in ['defpoint_x', 'defpoint_y', 'text_midpoint_x', 'text_midpoint_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
        for feat_name in ['main_value', 'tolerance']:
            col = feature_index_map[feat_name] + 1
            if entity_row[col] >= 0.0:
                entity_row[col] = random_shift_float(entity_row[col], shift_range=0.03)

    # LEADER
    elif entity_type == 'LEADER':
        for feat_name in ['vertices_x', 'vertices_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)

    # INSERT
    elif entity_type == 'INSERT':
        for feat_name in ['insert_insert_point_x', 'insert_insert_point_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
        for feat_name in ['scale_x', 'scale_y']:
            col = feature_index_map[feat_name] + 1
            if entity_row[col] >= 0.0:
                delta = np.random.uniform(-0.2, 0.2)
                entity_row[col] = max(0.0, entity_row[col] + delta)
        col_rot = feature_index_map['insert_rotation'] + 1
        entity_row[col_rot] = random_angle_shift(entity_row[col_rot], shift_range=0.1)

    # SPLINE
    elif entity_type == 'SPLINE':
        for feat_name in ['control_points_x', 'control_points_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)
        col_knots = feature_index_map['avg_knots'] + 1
        if entity_row[col_knots] >= 0.0:
            factor = np.random.uniform(0.8, 1.2)
            entity_row[col_knots] = max(0.0, entity_row[col_knots] * factor)

    # SOLID
    elif entity_type == 'SOLID':
        for feat_name in ['solid_points_x', 'solid_points_y']:
            col = feature_index_map[feat_name] + 1
            entity_row[col] = random_shift_float(entity_row[col], shift_range=0.05)

    return entity_row

def shuffle_entities(entities):

    n = len(entities)
    num_swaps = n // 2
    for _ in range(num_swaps):
        i = np.random.randint(0, n-1)
        entities[i], entities[i+1] = entities[i+1].copy(), entities[i].copy()
    return entities

def random_delete(entities, delete_ratio=0.1):

    n = len(entities)
    keep_mask = np.random.rand(n) > delete_ratio
    return entities[keep_mask]

def random_duplicate(entities, duplicate_ratio=0.1):

    n = len(entities)
    num_dup = int(n * duplicate_ratio)
    if num_dup <= 0:
        return entities
    chosen_indices = np.random.choice(np.arange(n), size=num_dup, replace=True)
    duplicates = []
    for idx in chosen_indices:
        new_entity = entities[idx].copy()
        new_entity = augment_entity_line(new_entity)
        duplicates.append(new_entity)
    augmented = np.concatenate([entities, np.array(duplicates, dtype=entities.dtype)], axis=0)
    return augmented

def fix_length_and_append_eos(entities, max_len=None):

    dim = len(FEATURE_NAMES) + 1
    out = np.full((max_len, dim), -1.0, dtype=np.float32)
    eos_idx = float(ENTITY_TYPES.index('EOS'))

    eos_line = np.full((dim,), -1.0, dtype=np.float32)
    eos_line[0] = eos_idx

    n = len(entities)
    if n >= max_len:
        out[:max_len-1, :] = entities[:max_len-1]
        out[max_len-1] = eos_line
    else:
        out[:n, :] = entities[:n]
        if n < max_len:
            out[n] = eos_line

    invalid_mask = (out[:, 0] < 0.0)
    out[invalid_mask, 0] = eos_idx

    return out


def augment_seq_sample(dxf_arr, do_shuffle=True, delete_ratio=0.1, duplicate_ratio=0.1):

    max_seq_len = dxf_arr.shape[0]
    eos_idx = float(ENTITY_TYPES.index('EOS'))

    valid_length = 0
    for i in range(max_seq_len):
        if dxf_arr[i, 0] == eos_idx:
            break
        valid_length += 1

    valid_entities = dxf_arr[:valid_length].copy()

    for row_idx in range(len(valid_entities)):
        valid_entities[row_idx] = augment_entity_line(valid_entities[row_idx])

    if do_shuffle and len(valid_entities) > 1:
        valid_entities = shuffle_entities(valid_entities)
    if delete_ratio > 0:
        valid_entities = random_delete(valid_entities, delete_ratio)
    if duplicate_ratio > 0:
        valid_entities = random_duplicate(valid_entities, duplicate_ratio)

    out = fix_length_and_append_eos(valid_entities, max_len=max_seq_len)
    return out


def augment_h5_dataset(input_h5_path, output_h5_path, shuffle=True, delete_ratio=0.1, duplicate_ratio=0.1):

    with h5py.File(input_h5_path, 'r') as fin, h5py.File(output_h5_path, 'w') as fout:
        if 'dxf_vec' not in fin:
            print(f"[Warning] {input_h5_path} no 'dxf_vec' skip")
            return
        dset_in = fin['dxf_vec']
        num_samples, seq_len, dim = dset_in.shape
        dset_out = fout.create_dataset(
            'dxf_vec',
            shape=(num_samples, seq_len, dim),
            dtype=np.float32
        )
        for i in range(num_samples):
            original_sample = dset_in[i]  # shape=(seq_len, dim), float32
            aug_sample = augment_seq_sample(
                original_sample,
                do_shuffle=shuffle,
                delete_ratio=delete_ratio,
                duplicate_ratio=duplicate_ratio
            )
            dset_out[i] = aug_sample
            if i % 50 == 0:
                print(f"  -> {input_h5_path}: Augmented sample {i}/{num_samples}")
    print(f"done : {input_h5_path} -> {output_h5_path}")


def augment_all_h5_in_directory(input_dir, output_dir,
                                shuffle=True, delete_ratio=0.1, duplicate_ratio=0.1):

    if not os.path.isdir(input_dir):
        raise ValueError(f"don't exist: {input_dir}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    all_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.h5')]
    if not all_files:
        print(f"{input_dir} no .h5。")
        return

    for h5_file in all_files:
        in_path = os.path.join(input_dir, h5_file)
        out_path = os.path.join(output_dir, h5_file)
        print(f"{in_path}")
        augment_h5_dataset(
            input_h5_path=in_path,
            output_h5_path=out_path,
            shuffle=shuffle,
            delete_ratio=delete_ratio,
            duplicate_ratio=duplicate_ratio
        )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Unquantized Seq data augmentation script")
    parser.add_argument("--input_dir", type=str, default=r"/path/to/continuous/h5")
    parser.add_argument("--output_dir", type=str, default=r"/path/to/augmented/h5")
    parser.add_argument("--delete_ratio", type=float, default=0.1)
    parser.add_argument("--duplicate_ratio", type=float, default=0.1)
    parser.add_argument("--disable_shuffle", action='store_true')
    args = parser.parse_args()

    do_shuffle = not args.disable_shuffle
    augment_all_h5_in_directory(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        shuffle=do_shuffle,
        delete_ratio=args.delete_ratio,
        duplicate_ratio=args.duplicate_ratio
    )
