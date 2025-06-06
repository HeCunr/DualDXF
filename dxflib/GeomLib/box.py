#dxflib/GeomLib/box.py
import ezdxf
import math
from ezdxf.entities import BoundaryPathType, EdgeType

def compute_entity_bounding_box(entity, doc):
    """
    Compute bounding box coordinates for a single entity, returns (min_x, min_y, max_x, max_y)
    """
    entity_type = entity.dxftype()
    points = []

    try:
        if entity_type == 'LINE':
            points = [entity.dxf.start, entity.dxf.end]

        elif entity_type == 'CIRCLE':
            center = entity.dxf.center
            radius = entity.dxf.radius
            min_x = center[0] - radius
            max_x = center[0] + radius
            min_y = center[1] - radius
            max_y = center[1] + radius
            return min_x, min_y, max_x, max_y

        elif entity_type == 'ARC':
            center = entity.dxf.center
            radius = entity.dxf.radius
            start_angle = math.radians(entity.dxf.start_angle)
            end_angle = math.radians(entity.dxf.end_angle)

            if end_angle < start_angle:
                end_angle += 2 * math.pi

            num_points = 100
            angles = [start_angle + t * (end_angle - start_angle) / (num_points - 1) for t in range(num_points)]
            points = [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]

        elif entity_type in ['TEXT', 'MTEXT']:
            insert_point = entity.dxf.insert
            height = entity.dxf.height if hasattr(entity.dxf, 'height') else 0
            width = entity.dxf.width if hasattr(entity.dxf, 'width') else 0
            x, y = insert_point[0], insert_point[1]
            min_x = x
            max_x = x + width
            min_y = y
            max_y = y + height
            return min_x, min_y, max_x, max_y

        elif entity_type == 'LWPOLYLINE':
            points = entity.get_points('xy')

        elif entity_type == 'POLYLINE':
            points = [vertex.dxf.location for vertex in entity.vertices]

        elif entity_type == 'SPLINE':
            if entity.has_fit_points:
                points = entity.fit_points
            else:
                points = entity.control_points

        elif entity_type == 'INSERT':
            insert_point = entity.dxf.insert
            x_scale = entity.dxf.xscale if hasattr(entity.dxf, 'xscale') else 1.0
            y_scale = entity.dxf.yscale if hasattr(entity.dxf, 'yscale') else 1.0
            rotation = entity.dxf.rotation if hasattr(entity.dxf, 'rotation') else 0.0

            block_name = entity.dxf.name
            block = doc.blocks.get(block_name)
            block_points = []

            for block_entity in block:
                bbox = compute_entity_bounding_box(block_entity, doc)
                if bbox:
                    min_x, min_y, max_x, max_y = bbox
                    block_points.extend([(min_x, min_y), (max_x, max_y)])

            if block_points:
                xs = [p[0] for p in block_points]
                ys = [p[1] for p in block_points]
                min_x_block = min(xs)
                max_x_block = max(xs)
                min_y_block = min(ys)
                max_y_block = max(ys)

                center_x = (min_x_block + max_x_block) / 2
                center_y = (min_y_block + max_y_block) / 2
                transformed_points = []

                for x, y in block_points:
                    x -= center_x
                    y -= center_y
                    x *= x_scale
                    y *= y_scale
                    angle = math.radians(rotation)
                    x_new = x * math.cos(angle) - y * math.sin(angle)
                    y_new = x * math.sin(angle) + y * math.cos(angle)
                    x_new += insert_point[0]
                    y_new += insert_point[1]
                    transformed_points.append((x_new, y_new))

                xs = [p[0] for p in transformed_points]
                ys = [p[1] for p in transformed_points]
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                return min_x, min_y, max_x, max_y
            else:
                return None

        elif entity_type == 'DIMENSION':
            dim_type = entity.dimtype
            dim_points = []

            def_point = entity.dxf.defpoint
            text_midpoint = entity.dxf.text_midpoint
            dim_points.append((def_point[0], def_point[1]))
            if text_midpoint:
                dim_points.append((text_midpoint[0], text_midpoint[1]))

            if hasattr(entity.dxf, 'dim_line_point'):
                dim_line_point = entity.dxf.dim_line_point
                dim_points.append((dim_line_point[0], dim_line_point[1]))

            if dim_type == 0:
                if hasattr(entity.dxf, 'defpoint2'):
                    def_point2 = entity.dxf.defpoint2
                    dim_points.append((def_point2[0], def_point2[1]))
                if hasattr(entity.dxf, 'defpoint3'):
                    def_point3 = entity.dxf.defpoint3
                    dim_points.append((def_point3[0], def_point3[1]))

            if dim_points:
                xs = [p[0] for p in dim_points]
                ys = [p[1] for p in dim_points]
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                return min_x, min_y, max_x, max_y
            else:
                return None

        elif entity_type == 'LEADER':
            points = entity.vertices

        elif entity_type == 'HATCH':
            for boundary_path in entity.paths:
                if boundary_path.type == BoundaryPathType.EDGE:
                    for edge in boundary_path.edges:
                        if edge.type == EdgeType.LINE:
                            points.append(edge.start)
                            points.append(edge.end)
                        elif edge.type == EdgeType.ARC:
                            center = edge.center
                            radius = edge.radius
                            start_angle = math.radians(edge.start_angle)
                            end_angle = math.radians(edge.end_angle)
                            if end_angle < start_angle:
                                end_angle += 2 * math.pi
                            num_points = 20
                            angles = [start_angle + t * (end_angle - start_angle) / (num_points - 1) for t in range(num_points)]
                            arc_points = [(center[0] + radius * math.cos(a), center[1] + radius * math.sin(a)) for a in angles]
                            points.extend(arc_points)
                        elif edge.type == EdgeType.ELLIPSE:
                            center = edge.center
                            major_axis = edge.major_axis
                            ratio = edge.radius_ratio
                            start_param = edge.start_param
                            end_param = edge.end_param
                            if end_param < start_param:
                                end_param += 2 * math.pi
                            num_points = 100
                            params = [start_param + t * (end_param - start_param) / (num_points - 1) for t in range(num_points)]
                            ellipse_points = []
                            for param in params:
                                cos_param = math.cos(param)
                                sin_param = math.sin(param)
                                x = center[0] + major_axis[0] * cos_param - major_axis[1] * sin_param * ratio
                                y = center[1] + major_axis[1] * cos_param + major_axis[0] * sin_param * ratio
                                ellipse_points.append((x, y))
                            points.extend(ellipse_points)
                        elif edge.type == EdgeType.SPLINE:
                            spline_points = edge.control_points
                            points.extend([(p[0], p[1]) for p in spline_points])
                        else:
                            pass
                elif boundary_path.type == BoundaryPathType.POLYLINE:
                    vertices = boundary_path.vertices
                    points.extend(vertices)
                else:
                    pass
        elif entity_type == 'SOLID':
            points = [
                (entity.dxf.vtx0.x, entity.dxf.vtx0.y),
                (entity.dxf.vtx1.x, entity.dxf.vtx1.y),
                (entity.dxf.vtx2.x, entity.dxf.vtx2.y)
            ]
        elif entity_type == 'SPLINE':
            points = [(p[0], p[1]) for p in entity.control_points]
        else:
            bbox = entity.bbox()
            if bbox:
                (min_x, min_y, _), (max_x, max_y, _) = bbox.extmin, bbox.extmax
                return min_x, min_y, max_x, max_y
            else:
                return None

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            return min_x, min_y, max_x, max_y
        else:
            return None

    except Exception as e:
        return None

def get_all_entities_bounding_boxes(dxf_file_path):
    doc = ezdxf.readfile(dxf_file_path)
    msp = doc.modelspace()
    entity_bounding_boxes = []

    for entity in msp:
        entity_type = entity.dxftype()
        if entity_type not in ['LINE', 'CIRCLE', 'ARC', 'LWPOLYLINE', 'TEXT',
                               'MTEXT', 'HATCH', 'DIMENSION', 'LEADER', 'INSERT',
                               'SPLINE', 'SOLID']:
            continue

        bbox = compute_entity_bounding_box(entity, doc)
        if bbox:
            min_x, min_y, max_x, max_y = bbox
            entity_bounding_boxes.append({
                'entity_type': entity_type,
                'handle': entity.dxf.handle,
                'min_x': min_x,
                'min_y': min_y,
                'max_x': max_x,
                'max_y': max_y
            })

    return entity_bounding_boxes

if __name__ == '__main__':
    dxf_file_path = r'C:\srtp\FIRST PAPER\encode\data\DeepDXF\TEST\DFN6BU(NiPdAu)-437 Rev1_1.dxf'
    bounding_boxes = get_all_entities_bounding_boxes(dxf_file_path)
    for bbox in bounding_boxes:
        if bbox['entity_type'] == 'SOILD':
            print(f"Entity Handle: {bbox['handle']}, Type: {bbox['entity_type']}, Bounding Box: (Min X: {bbox['min_x']}, Min Y: {bbox['min_y']}) - (Max X: {bbox['max_x']}, Max Y: {bbox['max_y']})")
