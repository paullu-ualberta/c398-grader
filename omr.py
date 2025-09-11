import cv2 as cv
import pymupdf
import numpy as np

from dataclasses import dataclass
from bisect import bisect_left
import logging
from itertools import chain

BGR_BLUE = (255, 0, 0)
BGR_GREEN = (0, 255, 0)

PDF_BLUE = (0.0, 0.0, 1.0)
PDF_GREEN = (0.0, 1.0, 0.0)

DPI = 200
TRIANGLE_MIN_AREA = DPI
NUM_OPTIONS = 6


@dataclass
class TransformationInfo:
    angle: float
    center: tuple[int, int]
    crop_top: int

    def translate_cords_back(self, x, y):
        # Convert angle to radians for math functions
        angle_rad = np.radians(-self.angle)  # Use negative angle for inverse rotation
        cos_angle = np.cos(angle_rad)
        sin_angle = np.sin(angle_rad)
        center_x, center_y = self.center

        # 1. Un-crop the y-coordinate
        y_uncropped = y + self.crop_top
        x_uncropped = x

        # 2. Un-rotate the point around the center
        # Translate point back to origin
        temp_x = x_uncropped - center_x
        temp_y = y_uncropped - center_y

        # Rotate point
        rotated_x = temp_x * cos_angle - temp_y * sin_angle
        rotated_y = temp_x * sin_angle + temp_y * cos_angle

        x_unrotated = rotated_x + center_x
        y_unrotated = rotated_y + center_y
        return x_unrotated, y_unrotated


@dataclass
class Bubble:
    x: int
    y: int
    radius: int

    @property
    def cords(self):
        return (self.x, self.y)

    def draw(self, img):
        cv.circle(img, self.cords, self.radius, BGR_GREEN)

    def to_pdf_cords(self, transform: TransformationInfo):
        assert transform is not None
        x_unrotated, y_unrotated = transform.translate_cords_back(self.x, self.y)

        # 3. Scale to PDF coordinates
        pdf_x = x_unrotated * 72 / DPI
        pdf_y = y_unrotated * 72 / DPI
        pdf_radius = int(self.radius * 72 / DPI)

        return Bubble(x=pdf_x, y=pdf_y, radius=pdf_radius)


@dataclass
class OrientationLine:
    start_x: int
    start_y: int
    end_x: int
    end_y: int

    @property
    def start_cords(self):
        return self.start_x, self.start_y

    @property
    def end_cords(self):
        return self.end_x, self.end_y

    @property
    def angle(self):
        return np.degrees(
            np.arctan2(self.end_y - self.start_y, self.end_x - self.start_x)
        )

    def draw(self, img):
        cv.line(img, self.start_cords, self.end_cords, BGR_BLUE, 2)


@dataclass
class GuideMark:
    x: int
    y: int
    width: int
    height: int

    @property
    def top_left_cords(self):
        return (self.x, self.y)

    @property
    def bottom_right_cords(self):
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

    def draw(self, img):
        cv.rectangle(img, self.top_left_cords, self.bottom_right_cords, BGR_BLUE)

    def to_pdf_cords(self, transform: TransformationInfo):
        assert transform is not None
        x_unrotated, y_unrotated = transform.translate_cords_back(self.x, self.y)

        # 3. Scale to PDF coordinates
        pdf_x = x_unrotated * 72 / DPI
        pdf_y = y_unrotated * 72 / DPI

        pdf_w = int(self.width * 72 / DPI)
        pdf_h = int(self.height * 72 / DPI)
        return GuideMark(x=pdf_x, y=pdf_y, width=pdf_w, height=pdf_h)


class GuideMatrix:
    def __init__(self, guide_points: list[GuideMark], tolerance=20):
        guide_points = sorted(guide_points, key=lambda g: g.y)

        # The vertical ones must be closest to the top.
        vertical_guides = guide_points[:NUM_OPTIONS]
        horizontal_guides = guide_points[NUM_OPTIONS:]
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


def get_line_angle(line):
    x1, y1, x2, y2 = line[0]
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))


def fix_page_orientation(page_img):
    img = cv.cvtColor(page_img, cv.COLOR_BGR2GRAY)

    sharpening_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv.filter2D(img, -1, sharpening_kernel)

    _, img = cv.threshold(img, 100, 255, cv.THRESH_BINARY_INV)

    erosion_kernel = np.ones((5, 5), np.uint8)
    img = cv.erode(img, erosion_kernel, iterations=1)

    contours, _ = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    h, w = page_img.shape[:2]
    min_line_length = int(w * 0.80)

    lines = []
    for contour in contours:
        _, _, c_w, c_h = cv.boundingRect(contour)
        if c_w > min_line_length and c_h < 50:  # Filter for long, horizontal contours
            # Fit a line to the contour points
            [vx, vy, x, y] = cv.fitLine(contour, cv.DIST_L2, 0, 0.01, 0.01)
            # Extrapolate the line to the image boundaries
            lefty = int((-x * vy / vx) + y)
            righty = int(((w - x) * vy / vx) + y)
            lines.append(
                OrientationLine(start_x=0, start_y=lefty, end_x=w - 1, end_y=righty)
            )

    if not lines:
        print("No dominant long lines were detected. Returning original image.")
        return page_img, None

    for line in lines:
        line.draw(page_img)

    angles = [line.angle for line in lines]
    median_angle = float(np.median(angles))
    print(f"Median angle found as {median_angle}")

    center = (w // 2, h // 2)
    rotation_matrix = cv.getRotationMatrix2D(center, median_angle, 1.0)
    rotated_image = cv.warpAffine(
        page_img,
        rotation_matrix,
        (w, h),
        flags=cv.INTER_CUBIC,
        borderMode=cv.BORDER_REPLICATE,
    )

    top = min(lines, key=lambda l: l.start_y).start_y
    bottom = max(lines, key=lambda l: l.start_y).start_y
    print(top, bottom)
    cropped_image = rotated_image[top:bottom, :]

    transformation_data = TransformationInfo(
        angle=float(median_angle), center=center, crop_top=top
    )
    return cropped_image, transformation_data


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
    print(guide_matricies[0].num_rows, guide_matricies[0].num_cols)
    answers = gather_into_columns(bubbles, column_cutoffs, key=lambda b: b.x)
    attempt_matrices = build_attempt_matrices(guide_matricies, answers)
    return attempt_matrices


def mark_single_page(page_image: cv.typing.MatLike):
    processed_image, transformation_data = fix_page_orientation(page_image)
    if transformation_data is None:
        # If orientation failed, we can't reliably detect anything.
        return [], [], [], None

    guides = detect_triangles(processed_image)
    bubbles = detect_bubbles(processed_image)

    print(f"Num guides: {len(guides)}")

    attempt_matrices = process_answers(processed_image, bubbles, guides)
    return list(chain(*attempt_matrices)), guides, bubbles, transformation_data


def get_image_from_page(page):
    page_image_bytes = page.get_pixmap(dpi=DPI).pil_tobytes(format="png")
    page_image = cv.imdecode(
        np.frombuffer(page_image_bytes, dtype=np.uint8), cv.IMREAD_UNCHANGED
    )
    assert page_image is not None
    return page_image


def get_answers_from_file(document):
    all_answers = []
    for page in document.pages():
        page_image = get_image_from_page(page)
        answers_on_this_page, _, _, _ = mark_single_page(page_image)
        all_answers.extend(answers_on_this_page)
    return all_answers


def draw_detected_objects_on_page(bubbles, guides, page):
    for bubble in bubbles:
        page.draw_circle(
            (bubble.x, bubble.y),
            bubble.radius,
            color=PDF_GREEN,
        )
    for guide in guides:
        page.draw_rect(
            (guide.x, guide.y, guide.x + guide.width, guide.y + guide.height),
            color=PDF_BLUE,
        )


def mark_file(attempt_file_bytes, answer_file_bytes):
    attempt_file = pymupdf.Document(stream=attempt_file_bytes)
    answer_file = pymupdf.Document(stream=answer_file_bytes)
    all_attempted_answers = []
    attempt_pages = list(attempt_file.pages())
    for page in attempt_pages:
        page_image = get_image_from_page(page)
        answer_attempts_on_this_page, guides, bubbles, transform_data = (
            mark_single_page(page_image)
        )
        all_attempted_answers.extend(answer_attempts_on_this_page)

        pdf_bubbles = [b.to_pdf_cords(transform_data) for b in bubbles]
        pdf_guides = [g.to_pdf_cords(transform_data) for g in guides]

        draw_detected_objects_on_page(pdf_bubbles, pdf_guides, page)

    answers = get_answers_from_file(answer_file)
    print("Answer:")
    for i, answer in enumerate(answers, start=1):
        print(f"{i:2}: {answer}")

    print("Attempt:")
    for i, attempted_answer in enumerate(all_attempted_answers, start=1):
        print(f"{i:2}: {attempted_answer}")

    final_score = sum(
        question_attempt == answer
        for (question_attempt, answer) in zip(all_attempted_answers, answers)
    )
    score_str = f"Score: {final_score}/{len(answers)}"
    print(score_str)

    first_page = attempt_pages[0]
    first_page.insert_text((20, 20), score_str, fontsize=24)

    ret = attempt_file.tobytes()
    attempt_file.close()
    return ret
