import cv2
import numpy as np
import csv



def normalize8(I):
    mn = I.min()
    mx = I.max()
    mx -= mn
    I = ((I - mn) / mx) * 255
    return I.astype(np.uint8)


def overlay_transparent(background, overlay, x, y):

    background_width = background.shape[1]
    background_height = background.shape[0]

    if x >= background_width or y >= background_height:
        return background

    h, w = overlay.shape[0], overlay.shape[1]

    if x + w > background_width:
        w = background_width - x
        overlay = overlay[:, :w]

    if y + h > background_height:
        h = background_height - y
        overlay = overlay[:h]

    if overlay.shape[2] < 4:
        overlay = np.concatenate(
            [
                overlay,
                np.ones((overlay.shape[0], overlay.shape[1], 1), dtype=overlay.dtype)
                * 255,
            ],
            axis=2,
        )

    overlay_image = overlay[..., :3]
    mask = overlay[..., 3:] / 255.0

    background[y : y + h, x : x + w] = (1.0 - mask) * background[
        y : y + h, x : x + w
    ] + mask * overlay_image

    return background


def get_mask_points(mask_name):
    if mask_name == "Front Man Mask":
        mask_path = "./assets/front_man_mask.png"
        csv_path = "./assets/front_man_mask.csv"
    elif mask_name == "Guards Mask":
        mask_path = "./assets/guards_mask.png"
        csv_path = "./assets/guards_mask.csv"
    elif mask_name == "Red Mask":
        mask_path = "./assets/redmask.png"
        csv_path = "./assets/red_mask.csv"
    elif mask_name == "Blue Mask":
        mask_path = "./assets/bluemask.png"
        csv_path = "./assets/blue_mask.csv"
    else:
        raise ValueError(f"❌ Unknown mask name: {mask_name}")

    mask_img = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
    if mask_img is None:
        raise FileNotFoundError(f"❌ Could not load mask image: {mask_path}")

    mask_img = mask_img.astype(np.float32) / 255.0

    mask_points = {}
    with open(csv_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")
        for i, row in enumerate(csv_reader):
            mask_points[int(row[0])] = [float(row[1]), float(row[2])]

    return mask_points, mask_img

def mask_overlay(image, faces, mask_up, mask_down, mask_name):
    mask_points, mask_img = get_mask_points(mask_name)
    mirror_point = {
        234: 1,
        93: 2,
        132: 3,
        58: 4,
        172: 5,
        136: 6,
        150: 7,
        149: 8,
        176: 9,
        148: 10,
        152: 11,
        377: 12,
        400: 13,
        378: 14,
        379: 15,
        365: 16,
        397: 17,
        288: 18,
        361: 19,
        323: 20,
        454: 21,
        356: 22,
        389: 23,
        251: 24,
        284: 25,
        332: 26,
        297: 27,
        338: 28,
        10: 29,
        109: 30,
        67: 31,
        103: 32,
        54: 33,
        21: 34,
        162: 35,
        127: 36,
    }
    
    mask_points = mask_points
    src_pts = []
    for i in sorted(mask_points.keys()):
        try:
            src_pts.append(np.array(mask_points[i]))
        except ValueError:
            continue
    src_pts = np.array(src_pts, dtype="float32")
    extend_y = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ]
    minimize_y = [22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36]
    face_points = {}
    for i in faces[0]:
        for j in mirror_point.keys():
            if i[0] == j:
                if mirror_point[i[0]] in minimize_y:
                    face_points[mirror_point[j]] = [
                        float(i[1]),
                        float(i[2] - int(mask_up)),
                    ]
                else:
                    if mirror_point[i[0]] in extend_y:
                        face_points[mirror_point[j]] = [
                            float(i[1]),
                            float(i[2] + int(mask_down)),
                        ]
                    else:
                        face_points[mirror_point[j]] = [float(i[1]), float(i[2])]
    dst_pts = []
    for i in sorted(face_points.keys()):
        try:
            dst_pts.append(np.array(face_points[i]))
        except ValueError:
            continue
    dst_pts = np.array(dst_pts, dtype="float32")
    M, _ = cv2.findHomography(src_pts, dst_pts)
    transformed_mask = cv2.warpPerspective(
        mask_img,
        M,
        (image.shape[1], image.shape[0]),
        None,
        cv2.INTER_LINEAR,
        cv2.BORDER_CONSTANT,
        borderValue=[0, 0, 0, 0],
    )
    png_image = normalize8(transformed_mask)
    new_image = overlay_transparent(image, png_image, 0, 0)
    # cv2.imwrite("output.png", new_image)
    return new_image
