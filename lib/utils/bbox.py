def uv2bbox(uv, bbox_scale=1.9):
    c_x = (min(uv[:, 0]) + max(uv[:, 0])) / 2
    c_y = (min(uv[:, 1]) + max(uv[:, 1])) / 2
    w = max(uv[:, 0] - min(uv[:, 0]))
    h = max(uv[:, 1] - min(uv[:, 1]))
    size = max(w, h) / 2 * bbox_scale # 1.3  1.9 for KETI2020
    
    tl_x = int(c_x - size)
    tl_y = int(c_y - size)
    br_x = int(c_x + size)
    br_y = int(c_y + size)

    return [tl_x, tl_y, br_x, br_y]

def get_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0
    return inter_area / union_area
