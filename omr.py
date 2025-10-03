import cv2 as cv
import pymupdf
import numpy as np

from dataclasses import dataclass
from bisect import bisect_left
import logging
from itertools import chain

BGR_BLUE = (255, 0, 0)
BGR_GREEN = (0, 255, 0)

PDF_RED = (1.0, 0.0, 0.0)
PDF_GREEN = (0.0, 1.0, 0.0)
PDF_BLUE = (0.0, 0.0, 1.0)

DPI = 200
TRIANGLE_MIN_AREA = DPI

logger = logging.getLogger("OMR")


@dataclass
class TransformationInfo:
    angle: float
    center: tuple
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

    def shifted_by(self, x_offset, y_offset):
        return GuideMark(
            x=self.x + x_offset,
            y=self.y + y_offset,
            width=self.width,
            height=self.height,
        )


def detect_horizontal_and_vertical_guides(all_guides, tolerance=10):
    # Since there are always going to be more questions than options
    # first we figure out the x coordinate of the horizontal guides
    assert len(all_guides) > 2
    horizontal_guide_x = sorted([guide.x for guide in all_guides])[len(all_guides) // 2]
    horizontal_guides = []
    vertical_guides = []
    for guide in all_guides:
        if abs(guide.x - horizontal_guide_x) < tolerance:
            horizontal_guides.append(guide)
        else:
            vertical_guides.append(guide)
    return vertical_guides, horizontal_guides


class GuideMatrix:
    def __init__(self, guide_points: list):
        v_guides, h_guides = detect_horizontal_and_vertical_guides(guide_points)
        logger.debug(f"Detected a grid {len(v_guides)}x{len(h_guides)} grid")
        self.vertical_guides = v_guides
        self.vertical_guides.sort(key=lambda g: g.x)
        self.horizontal_guides = h_guides
        self.horizontal_guides.sort(key=lambda g: g.y)

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

    def to_pdf_cords(self, transform: TransformationInfo):
        pdf_guides = [g.to_pdf_cords(transform) for g in self.vertical_guides] + [
            g.to_pdf_cords(transform) for g in self.horizontal_guides
        ]
        return GuideMatrix(pdf_guides)

    def horizontal_guide_for_row(self, row):
        return self.horizontal_guides[row]

    def __repr__(self):
        return f"GuideMatrix<{self.num_rows}x{self.num_cols}>(<{self.horizontal_guides}>x<{self.vertical_guides}>)"


def show_image(img, convert_from_grayscale=False):
    if convert_from_grayscale:
        img = cv.cvtColor(img, cv.COLOR_GRAY2RGB)
    cv.imshow("Image", img)
    cv.waitKey(0)


def preprocess_image_for_detection(
    img: cv.typing.MatLike, /, blur_mask=5, threshold=230
) -> cv.typing.MatLike:
    if blur_mask > 3:
        img = cv.GaussianBlur(img, (blur_mask, blur_mask), 0)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, threshold, 255, cv.THRESH_BINARY_INV)
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
        logger.warning(
            "No dominant long lines were detected. Returning original image."
        )
        return page_img, None

    for line in lines:
        line.draw(page_img)

    angles = [line.angle for line in lines]
    median_angle = float(np.median(angles))
    logger.debug(f"Rotating image by {-median_angle} degrees")

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
    logger.debug(f"Cropping image to height: {top}, {bottom}")
    cropped_image = rotated_image[top:bottom, :]

    transformation_data = TransformationInfo(
        angle=float(median_angle), center=center, crop_top=top
    )
    return cropped_image, transformation_data


def detect_triangles(img, min_area=DPI + 50):
    # Need to use a lower threshold here because students sometimes squible light
    # marks around the triangles.
    img = preprocess_image_for_detection(img, threshold=180)
    # show_image(img)
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
    min_area=DPI,
    circularity_threshold=0.8,
    solidity_threshold=0.8,
):
    # Need big blur mask and threshold because students sometimes don't pencil
    # in the mark enough or the scanner makes their marks look jagged.
    img = preprocess_image_for_detection(img, blur_mask=17)
    if False:
        erosion_kernel = np.ones((11, 11), np.uint8)
        img = cv.erode(img, erosion_kernel, iterations=1)
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


def has_a_bubble_at(point, bubbles, tolerance=30):
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
    for guide_matrix, bubbles in zip(guide_matrices, bubble_columns):
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


def get_attempt_matrix_from_raw_objs(
    img: cv.typing.MatLike, bubbles: list, guide_points
):
    column_cutoffs = [int(img.shape[1] / 2)]  # Only 2 columns for now
    columns = gather_into_columns(guide_points, column_cutoffs, key=lambda g: g.x)
    guide_matricies = [GuideMatrix(c) for c in columns]
    answers = gather_into_columns(bubbles, column_cutoffs, key=lambda b: b.x)
    attempt_matrices = build_attempt_matrices(guide_matricies, answers)
    return list(chain(*attempt_matrices)), guide_matricies


def draw_all_objects_on(to_draw_on, *objs_to_draw):
    for obj in objs_to_draw:
        obj.draw(to_draw_on)


def get_attempts_on_page_img(page_image: cv.typing.MatLike):
    processed_image, transformation_data = fix_page_orientation(page_image)
    if transformation_data is None:
        # If orientation failed, we can't reliably detect anything.
        return [], [], [], [], None

    guides = detect_triangles(processed_image)
    bubbles = detect_bubbles(processed_image)

    # draw_all_objects_on(processed_image, *guides, *bubbles)
    # show_image(processed_image)
    logger.debug(f"Detected {len(guides)} guides")

    attempt_matrix, guide_matrices = get_attempt_matrix_from_raw_objs(
        processed_image, bubbles, guides
    )
    return attempt_matrix, guides, bubbles, guide_matrices, transformation_data


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
        answers_on_this_page, *_ = get_attempts_on_page_img(page_image)
        all_answers.append(answers_on_this_page)
    return all_answers


def draw_correct_answers_on_page(guide_matrices, answers, radius, page):
    guide_matrices = list(guide_matrices)
    num_rows_in_matrix = guide_matrices[0].num_rows
    rot_matrix = page.derotation_matrix
    for row, answer in enumerate(answers):
        guide_matrix = guide_matrices[row // num_rows_in_matrix]
        for col, option in enumerate(answer):
            if option != 1:
                continue
            center = pymupdf.Point(
                guide_matrix.cell_center_at(row % num_rows_in_matrix, col)
            )
            center = center * rot_matrix
            page.draw_circle(center, radius, color=PDF_GREEN)


def draw_question_status_on_page(guide_matrices, results, page):
    guide_matrices = list(guide_matrices)
    num_rows_in_matrix = guide_matrices[0].num_rows
    rot_matrix = page.derotation_matrix
    for row, result in enumerate(results):
        guide_matrix = guide_matrices[row // num_rows_in_matrix]
        guide = guide_matrix.horizontal_guide_for_row(row % num_rows_in_matrix)
        guide = guide.shifted_by(-guide.width - 2, 0)
        rect = pymupdf.Rect(
            guide.x, guide.y, guide.x + guide.width, guide.y + guide.height
        )
        rect = rect * rot_matrix
        rect.normalize()
        color = PDF_GREEN if result else PDF_RED
        page.draw_rect(rect, color=color, fill=color)


def draw_detected_objects_on_page(bubbles, guides, page):
    rot_matrix = page.derotation_matrix
    for bubble in bubbles:
        center = pymupdf.Point(bubble.x, bubble.y)
        center = center * rot_matrix
        page.draw_circle(
            center,
            bubble.radius,
            color=PDF_RED,
        )

    for guide in guides:
        rect = pymupdf.Rect(
            guide.x, guide.y, guide.x + guide.width, guide.y + guide.height
        )
        rect = rect * rot_matrix
        rect.normalize()
        page.draw_rect(rect, color=PDF_BLUE)


def correct_attempt_positions(answer_matrix, attempt_matrix):
    assert len(attempt_matrix) == len(answer_matrix)
    assert len(attempt_matrix[0]) == len(answer_matrix[0])
    positions = []
    for attempt, answer in zip(attempt_matrix, answer_matrix):
        # for now, the first question with no correct answers marks, the end
        # of the questions
        if not any(answer):
            break
        positions.append(answer == attempt)
    return positions


def calculate_final_score(attempt_matrix, answer_matrix):
    positions = correct_attempt_positions(answer_matrix, attempt_matrix)
    return sum(positions), len(positions)


def mark_pages(attempt_pages, answers):
    all_attempts = []
    for page, correct_answers_on_this_page in zip(attempt_pages, answers):
        page_image = get_image_from_page(page)
        attempts, guides, bubbles, guide_matrices, transform_data = (
            get_attempts_on_page_img(page_image)
        )
        assert transform_data is not None
        all_attempts.extend(attempts)

        pdf_bubbles = [b.to_pdf_cords(transform_data) for b in bubbles]
        pdf_guides = [g.to_pdf_cords(transform_data) for g in guides]
        pdf_guide_matrices = [gm.to_pdf_cords(transform_data) for gm in guide_matrices]

        logger.debug(f"The PDF page is rotated: {page.rotation} degrees")
        draw_detected_objects_on_page(pdf_bubbles, pdf_guides, page)
        draw_correct_answers_on_page(
            pdf_guide_matrices,
            correct_answers_on_this_page,
            pdf_bubbles[0].radius + 3,
            page,
        )
        correct_positions = correct_attempt_positions(
            correct_answers_on_this_page, attempts
        )
        draw_question_status_on_page(pdf_guide_matrices, correct_positions, page)

    answers = list(chain(*answers))
    logger.info("Found Answer Matrix as:")
    for i, answer in enumerate(answers, start=1):
        logger.info(f" {i:2}: {answer}")

    logger.info("Attempt:")
    for i, attempted_answer in enumerate(all_attempts, start=1):
        logger.info(f" {i:2}: {attempted_answer}")

    score, total_answers = calculate_final_score(all_attempts, answers)
    score_str = f"Score: {score}/{total_answers}"
    logger.info(f"Score: {score_str}")

    first_page = attempt_pages[0]
    score_loc = pymupdf.Point(20, 20)
    score_loc = score_loc * first_page.derotation_matrix
    first_page.insert_text(
        score_loc, score_str, fontsize=24, rotate=first_page.rotation
    )
    return score, total_answers


def chunked(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def mark_single_file(attempt_file_bytes, answer_file_bytes):
    attempt_file = pymupdf.Document(stream=attempt_file_bytes)
    answer_file = pymupdf.Document(stream=answer_file_bytes)
    logger.debug("Start processing answer file")
    answers = get_answers_from_file(answer_file)
    attempt_pages = list(attempt_file.pages())
    assert len(attempt_pages) % len(answers) == 0, (
        "Not all attempts seem to have all pages"
    )
    attempts = chunked(attempt_pages, len(answers))
    for attempt in attempts:
        mark_pages(attempt, answers)
    ret = attempt_file.tobytes()
    attempt_file.close()
    return ret


def mark_file(attempt_file_bytes, answer_file_bytes):
    attempt_file = pymupdf.Document(stream=attempt_file_bytes)
    answer_file = pymupdf.Document(stream=answer_file_bytes)
    logger.debug("Start processing answer file")
    answers = get_answers_from_file(answer_file)
    logger.debug("Start processing attempt file")
    attempt_pages = list(attempt_file.pages())
    score, total_answers = mark_pages(attempt_pages, answers)
    ret = attempt_file.tobytes()
    attempt_file.close()
    return score, total_answers, ret
