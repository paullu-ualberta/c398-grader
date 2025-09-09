import cv2 as cv
import pymupdf
import numpy as np

from dataclasses import dataclass
from bisect import bisect_left
import logging
from itertools import chain

BGR_BLUE = (255, 0, 0)
BGR_GREEN = (0, 255, 0)

DPI = 200
TRIANGLE_MIN_AREA = DPI


@dataclass
class Bubble:
    x: int
    y: int
    radius: int

    @property
    def cords(self):
        return (self.x, self.y)


@dataclass
class GuideMark:
    x: int
    y: int
    width: int
    height: int

    @property
    def upper_left_cords(self):
        return (self.x, self.y)

    @property
    def lower_right_cords(self):
        return (self.x + self.width, self.y + self.height)

    @property
    def center_cords(self):
        return (self.center_x, self.center_y)

    @property
    def center_x(self):
        return self.x + self.width / 2

    @property
    def center_y(self):
        return self.y + self.height / 2


class GuideMatrix:
    def __init__(self, guide_points: list[GuideMark], tolerance=20):
        guide_points = sorted(guide_points, key=lambda g: g.y)
        vertical_guides = guide_points[
            :5
        ]  # The vertical ones must be closest to the top.
        horizontal_guides = guide_points[5:]
        self.vertical_guides = vertical_guides
        self.vertical_guides.sort(key=lambda g: g.x)
        self.horizontal_guides = horizontal_guides
        self.tolerance = tolerance

    def cells_centers(self):
        for vert_guide in self.vertical_guides:
            for horizontal_guide in self.horizontal_guides:
                yield (vert_guide.center_x, horizontal_guide.center_y)

    @property
    def num_rows(self):
        return len(self.horizontal_guides)

    @property
    def num_cols(self):
        return len(self.vertical_guides)

    def cell_center_at(self, row, col):
        vert_guide = self.vertical_guides[col]
        hor_guide = self.horizontal_guides[row]
        return (vert_guide.center_x, hor_guide.center_y)


def show_image(img, convert_from_grayscale=False):
    if convert_from_grayscale:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.imshow("Image", img)
    cv.waitKey(0)


def preprocess_image_for_detection(
    img: cv.typing.MatLike, /, blur_mask=5
) -> cv.typing.MatLike:
    if blur_mask > 3:
        img = cv.GaussianBlur(img, (blur_mask, blur_mask), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 230, 255, cv.THRESH_BINARY_INV)
    return img


def fix_page_orientation(page_img):
    # Get image dimensions
    h, w = page_img.shape[:2]

    # 1. Pre-processing for Thick Line Isolation
    img = cv.cvtColor(page_img, cv.COLOR_BGR2GRAY)

    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv.filter2D(img, -1, sharpening_kernel)

    _, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)

    # Erode the image to remove noise and thin lines
    erosion_kernel = np.ones((5, 5), np.uint8)
    img = cv.erode(img, erosion_kernel, iterations=1)
    show_image(img)

    # Find contours instead of using Canny edge detection
    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    lines_list = []
    min_line_length = int(w * 0.80)
    for contour in contours:
        x_c, y_c, w_c, h_c = cv.boundingRect(contour)
        if w_c > min_line_length and h_c < 50:  # Filter for long, horizontal contours
            # Fit a line to the contour points
            [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
            # Extrapolate the line to the image boundaries
            lefty = int((-x * vy / vx) + y)
            righty = int(((w - x) * vy / vx) + y)
            lines_list.append([[0, lefty, w - 1, righty]])

    if not lines_list:
        print("No dominant long lines were detected. Returning original image.")
        return page_img
    lines = np.array(lines_list)

    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(page_img, (x1, y1), (x2, y2), BGR_BLUE, 2)
    show_image(page_img)

    # 3. Angle Calculation (same as before)
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        angles.append(angle)

    median_angle = np.median(angles)
    print(f"Median angle found as {median_angle}")

    # 4. Image Rotation (same as before)
    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, median_angle, 1.0)
    rotated_image = cv.warpAffine(
        page_img,
        rotation_matrix,
        (w, h),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE,
    )

    top = min(lines, key=lambda l: l[0][1])[0][1]
    bottom = max(lines, key=lambda l: l[0][1])[0][1]
    print(top, bottom)
    cropped_image = rotated_image[top : top + bottom, :]
    show_image(cropped_image)

    return cropped_image


def detect_triangles(img, min_area=DPI + 50):
    img = preprocess_image_for_detection(img, 1)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    all_triangles = []

    for i, contour in enumerate(contours):
        # Filter out very small contours that might be noise
        if cv.contourArea(contour) < min_area:
            continue

        # Approximate the contour to a polygon.
        # The epsilon parameter is key; it determines how "closely" the
        # polygon must match the contour. A smaller value means a closer match.
        # We use a percentage of the contour's perimeter.
        epsilon = 0.04 * cv.arcLength(contour, True)
        approx = cv.approxPolyDP(contour, epsilon, True)

        # A triangle will have 3 vertices
        if len(approx) != 3:
            continue
        # Check if it's a filled shape by ensuring it has no child contours.
        # hierarchy[0][i][2] is the index of the first child contour.
        # It's -1 if there are no children.
        if hierarchy[0][i][2] != -1:
            continue
        x, y, w, h = cv.boundingRect(contour)
        all_triangles.append(GuideMark(x=int(x), y=int(y), width=int(w), height=int(h)))

    return all_triangles


def detect_bubbles(
    img,
    min_area=50,
    circularity_threshold=0.8,
    solidity_threshold=0.8,
):
    img = preprocess_image_for_detection(img, blur_mask=17)
    contours, hierarchy = cv.findContours(img, cv.RETR_CCOMP, cv.CHAIN_APPROX_SIMPLE)
    detected_circles = []
    for i, contour in enumerate(contours):
        # We are looking for contours that are external (no parent) and have no children.
        # This indicates a solid, filled shape.
        # hierarchy[0][i][3] is the parent index. -1 means no parent (i.e., external).
        # hierarchy[0][i][2] is the first child index. -1 means no child.
        child = hierarchy[0][i][2]
        parent = hierarchy[0][i][3]
        is_filled_shape = parent == -1 and child == -1

        if not is_filled_shape:
            logging.info("Skipping because its not a filled image")
            continue

        # This contour is a candidate for a filled shape.
        # Now, apply the geometric filters to see if it's a circle.

        # a) Filter by area to remove small noise
        area = cv.contourArea(contour)
        if area < min_area:
            logging.info("Skipping because area less than min area")
            continue

        # b) Calculate circularity to filter non-circular shapes
        perimeter = cv.arcLength(contour, True)
        if perimeter == 0:
            logging.info("Skipping because parameter is 0")
            continue
        circularity = (4 * np.pi * area) / (perimeter * perimeter)

        if circularity < circularity_threshold:
            logging.info(
                f"Skipping because circularity is less than threshold {circularity}vs{circularity_threshold}"
            )
            continue

        # c) The solidity check is now a secondary check for convexity.
        # Solidity = Contour Area / Convex Hull Area
        # A filled circle will have a solidity close to 1.
        hull = cv.convexHull(contour)
        hull_area = cv.contourArea(hull)
        if hull_area == 0:
            logging.info("Skipping because hull area is zero")
            continue

        solidity = float(area) / hull_area
        if solidity < solidity_threshold:
            logging.info(
                f"Skipping because solidity is less than threshold {solidity}vs{solidity_threshold}"
            )
            continue

        # This contour is a good candidate for a filled circle
        # Get the enclosing circle
        ((x, y), radius) = cv.minEnclosingCircle(contour)
        detected_circles.append(Bubble(x=int(x), y=int(y), radius=int(radius)))
    return detected_circles


def gather_into_columns(lst, demarkation_points, key=lambda x: x):
    assert sorted(demarkation_points) == demarkation_points, (
        "demarkation points must be sorted"
    )
    columns = [[] for _ in range(len(demarkation_points) + 1)]
    for item in lst:
        idx = bisect_left(demarkation_points, key(item))
        columns[idx].append(item)
    return columns


def get_matrices(guide_points, column_cutoffs):
    columns = gather_into_columns(guide_points, column_cutoffs, key=lambda g: g.x)
    return [GuideMatrix(c) for c in columns]


def has_a_bubble_at(point, bubbles, tolerance=100):
    for bubble in bubbles:
        if (
            abs(point[0] - bubble.x) < tolerance
            and abs(point[1] - bubble.y) < tolerance
        ):
            return True
    return False


def build_attempt_matrices(guide_matrices, bubble_columns):
    assert len(guide_matrices) == len(bubble_columns), (
        "Both guide matrices and answers must be the same size"
    )
    attempt_matrices = []
    for col_idx in range(len(guide_matrices)):
        guide_matrix = guide_matrices[col_idx]
        bubbles = bubble_columns[col_idx]
        num_rows = guide_matrix.num_rows
        num_cols = guide_matrix.num_cols
        attempt_matrix = [[0 for _ in range(num_cols)] for _ in range(num_rows)]

        for row in range(num_rows):
            for col in range(num_cols):
                cell_center = guide_matrix.cell_center_at(row, col)
                if has_a_bubble_at(cell_center, bubbles):
                    attempt_matrix[row][col] = 1
        attempt_matrices.append(attempt_matrix)
    return attempt_matrices


def process_answers(img: cv.typing.MatLike, bubbles: list[Bubble], guide_points):
    column_cutoffs = [int(img.shape[1] / 2)]  # Only 2 columns for now
    guide_matricies = get_matrices(guide_points, column_cutoffs)
    answers = gather_into_columns(bubbles, column_cutoffs, key=lambda b: b.x)
    attempt_matrices = build_attempt_matrices(guide_matricies, answers)
    return attempt_matrices


def mark_single_page(page_image: cv.typing.MatLike):
    page_image = fix_page_orientation(page_image)
    guides = detect_triangles(page_image)
    bubbles = detect_bubbles(page_image)
    for circle in bubbles:
        cv.circle(page_image, circle.cords, circle.radius, BGR_GREEN, 0)

    for guide in guides:
        cv.rectangle(
            page_image, guide.upper_left_cords, guide.lower_right_cords, BGR_BLUE, 0
        )
    print(f"Num guides: {len(guides)}")

    attempt_matrices = process_answers(page_image, bubbles, guides)
    return list(chain(*attempt_matrices))


def get_answers_from_file(document):
    all_answers = []
    pages = list(document.pages())
    assert len(pages) > 0, "PDF has zero pages"
    for page in pages:
        page_image_bytes = page.get_pixmap(dpi=DPI).pil_tobytes(format="png")
        page_image = cv.imdecode(
            np.frombuffer(page_image_bytes, dtype=np.uint8), cv.IMREAD_UNCHANGED
        )
        assert page_image is not None
        answers_on_this_page = mark_single_page(page_image)
        all_answers.extend(answers_on_this_page)

    for row_num, row in enumerate(all_answers, start=1):
        print(f"{row_num:2}: {row}")
    return all_answers


def mark_file(attempt_file, answer_file):
    attempt_file = pymupdf.Document(stream=attempt_file)
    answer_file = pymupdf.Document(stream=answer_file)
    print("Answer:")
    answers = get_answers_from_file(answer_file)
    print("Attempt:")
    attempt = get_answers_from_file(attempt_file)
    final_score = sum(
        question_attempt == answer
        for (question_attempt, answer) in zip(attempt, answers)
    )
    score_str = f"Score: {final_score}/{len(answers)}"
    first_page = next(attempt_file.pages())
    first_page.insert_text((20, 20), score_str, fontsize=24)
    print(score_str)
    return attempt_file.tobytes()
