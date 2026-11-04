"""Current-image recovery contracts; synthetic images and camera poses only."""
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from peleng.geometry import CameraRig

try:
    from peleng.recovery import ImageEvidence
except ModuleNotFoundError:
    ImageEvidence = None


def make_rig(directory, *, moving=False):
    text = ('Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;10\n'
            '0;0;0;100;0;0;0;;;matrix;40*22.5mm\n')
    if moving:
        text += '2;0;0;100;0;0;5;;;;\n'
    (Path(directory) / 'object_transform_CineCameraActor1.csv').write_text(text, encoding='cp1251')
    return CameraRig(directory, width=320, height=180)


def scene(x=None):
    image = np.full((180, 320, 3), (30, 45, 60), dtype=np.uint8)
    # Fixed detail distinguishes background from the moving target.
    cv2.rectangle(image, (240, 20), (285, 70), (70, 85, 95), -1)
    if x is not None:
        cv2.rectangle(image, (x, 95), (x+39, 124), (195, 205, 215), -1)
        cv2.line(image, (x+10, 95), (x+10, 124), (80, 85, 90), 2)
    return image


def neural(x, confidence=.8):
    return {'bbox': [x, 95, x+40, 125], 'confidence': confidence, 'label': 'truck'}


def step(evidence, image, n, boxes=(), timestamp=None):
    assert evidence is not None, 'Causal image recovery has not been implemented'
    return evidence.update([image], [1], n, n/30 if timestamp is None else timestamp,
                           [list(boxes)])[0]


def foreground(candidates):
    return [box for box in candidates if box['source'] == 'foreground' and box['measured']]


@pytest.fixture
def evidence(tmp_path):
    return ImageEvidence(make_rig(tmp_path)) if ImageEvidence is not None else None


def seed(evidence):
    step(evidence, scene(), 0)
    step(evidence, scene(40), 1, [neural(40)])


def test_current_pixels_continue_target_after_neural_miss(evidence):
    # Fails if recovery returns remembered/predicted boxes instead of current pixels.
    seed(evidence)
    found = foreground(step(evidence, scene(48), 2))
    assert len(found) == 1
    box = found[0]
    assert np.allclose(box['bbox'], [48, 95, 88, 125], atol=2)
    assert box['neural_detected'] is False
    assert box['quality']['seed_frame'] == 1
    assert 0 < box['confidence'] <= .5


def test_current_neural_is_distinct_and_can_seed_during_background_warmup(evidence):
    boxes = step(evidence, scene(40), 0, [neural(40)])
    assert len(boxes) == 1
    assert boxes[0]['source'] == 'neural'
    assert boxes[0]['neural_detected'] is True
    assert boxes[0]['measured'] is True


def test_unseeded_motion_is_not_a_target_measurement(evidence):
    step(evidence, scene(), 0)
    assert foreground(step(evidence, scene(40), 1)) == []
    assert foreground(step(evidence, scene(48), 2)) == []


def test_disappearance_stops_current_image_measurement(evidence):
    seed(evidence)
    assert foreground(step(evidence, scene(48), 2))
    assert foreground(step(evidence, scene(), 3)) == []
    assert foreground(step(evidence, scene(), 4)) == []


def test_remote_weak_static_detection_cannot_seed_or_replace_moving_target(evidence):
    step(evidence, scene(), 0)
    rock = {'bbox': [240, 20, 286, 71], 'confidence': .26, 'label': 'truck'}
    for n in range(1, 5):
        boxes = step(evidence, scene(40+n*3), n, [rock])
        assert not any(box['source'] == 'neural' for box in boxes)
        assert foreground(boxes) == []


def test_long_gap_discards_old_identity(evidence):
    seed(evidence)
    assert foreground(step(evidence, scene(48), 2))
    assert foreground(step(evidence, scene(51), 100)) == []


def test_short_skipped_frames_still_use_current_image_box(evidence):
    seed(evidence)
    found = foreground(step(evidence, scene(60), 7))
    assert len(found) == 1
    assert np.allclose(found[0]['bbox'], [60, 95, 100, 125], atol=2)


def test_reset_removes_previous_replay_identity(evidence):
    seed(evidence)
    evidence.reset()
    assert step(evidence, scene(), 0) == []
    assert foreground(step(evidence, scene(48), 1)) == []


def test_rotating_camera_does_not_measure_scene_change_as_target(tmp_path):
    assert ImageEvidence is not None
    recovery = ImageEvidence(make_rig(tmp_path, moving=True))
    seed(recovery)
    shifted = cv2.warpAffine(scene(), np.float32([[1, 0, 12], [0, 1, 0]]), (320, 180))
    assert foreground(step(recovery, shifted, 2)) == []


def test_inputs_reject_nonfinite_time_without_poisoning_state(evidence):
    seed(evidence)
    for timestamp in [math.nan, math.inf, -math.inf]:
        with pytest.raises(ValueError):
            step(evidence, scene(48), 2, timestamp=timestamp)
    assert foreground(step(evidence, scene(48), 2))


def test_backwards_replay_requires_explicit_reset(evidence):
    seed(evidence)
    with pytest.raises(ValueError):
        step(evidence, scene(48), 0)


def test_bad_neural_boxes_are_ignored(evidence):
    invalid = [None, {}, neural(40, math.nan), neural(40, 1.5),
               {'bbox': [1, 2, 1, 9], 'confidence': .8}]
    assert step(evidence, scene(), 0, invalid) == []


def test_append_future_frames_cannot_change_prior_outputs(tmp_path):
    assert ImageEvidence is not None
    rig = make_rig(tmp_path)
    first, second = ImageEvidence(rig), ImageEvidence(rig)
    a, b = [], []
    for n, x in enumerate([None, 40, 48, 56]):
        boxes = [neural(40)] if n == 1 else []
        a.append(step(first, scene(x), n, boxes))
        b.append(step(second, scene(x), n, boxes))
    for n in range(4, 9):
        step(first, scene(200), n)
        step(second, scene(), n)
    assert a == b


def test_removed_long_stationary_target_does_not_leave_background_ghost(evidence):
    for n in range(181):
        step(evidence, scene(40), n, [neural(40)])
    assert foreground(step(evidence, scene(), 181)) == []


def test_fuller_current_silhouette_replaces_partial_neural_duplicate(evidence):
    seed(evidence)
    partial = {'bbox': [48, 95, 63, 125], 'confidence': .7, 'label': 'truck'}
    boxes = step(evidence, scene(48), 2, [partial])
    assert len(boxes) == 1
    assert boxes[0]['source'] == 'foreground'
    assert np.allclose(boxes[0]['bbox'], [48, 95, 88, 125], atol=2)


def test_foreground_shadow_does_not_override_complete_current_neural_box(evidence):
    seed(evidence)
    image = scene(48)
    cv2.rectangle(image, (48, 125), (87, 134), (10, 15, 20), -1)
    boxes = step(evidence, image, 2, [neural(48)])
    assert len(boxes) == 1
    assert boxes[0]['source'] == 'neural'
    assert boxes[0]['bbox'] == [48, 95, 88, 125]


def pan_fixture(directory, *, translating=False):
    text = ('Frame;Loc_X;Loc_Y;Loc_Z;Rot_Roll;Rot_Pitch;Rot_Yaw;;;Focal length;40\n'
            '0;0;0;100;0;0;0;;;matrix;40*22.5mm\n')
    for n in range(3, 31):
        text += f'{n};{10 if translating else 0};0;100;0;0;{-(n-2)*.6};;;;\n'
    (Path(directory)/'object_transform_CineCameraActor1.csv').write_text(text, encoding='cp1251')
    rig = CameraRig(directory, width=320, height=180)
    # Project directions independently of the recovery homography helper.
    corners = np.float32([[0, 0], [319, 0], [319, 179], [0, 179]])
    matrices = {}
    for n in range(31):
        projected = []
        for pixel in corners:
            origin, ray = rig.ray(1, 0, pixel)
            projected.append(rig.project(1, n, origin+1_000_000*ray))
        matrices[n] = cv2.getPerspectiveTransform(corners, np.float32(projected))
    rng = np.random.default_rng(281)
    texture = rng.integers(25, 100, (180, 320, 3), dtype=np.uint8)
    texture = cv2.GaussianBlur(texture, (7, 7), 1.5)
    def image(n, x=None):
        bg = cv2.warpPerspective(texture, matrices[n], (320, 180),
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(55, 55, 55))
        if x is not None:
            cv2.rectangle(bg, (x, 95), (x+39, 124), (195, 205, 215), -1)
            cv2.line(bg, (x+10, 95), (x+10, 124), (80, 85, 90), 2)
        return bg
    return rig, image


def test_panning_camera_keeps_full_current_silhouette_after_neural_gap(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    for n in range(3, 21):
        x = 44+3*(n-2)
        found = foreground(step(recovery, image(n, x), n))
        assert len(found) == 1, f'Lost current silhouette on pan frame {n}'
        assert np.allclose(found[0]['bbox'], [x, 95, x+40, 125], atol=3)
        assert found[0]['quality']['measurement'] == 'rotation_compensated_background'


def test_pure_camera_pan_cannot_turn_static_scene_into_target(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    # Even an old erroneous seed must not make newly panning texture measured.
    step(recovery, image(0), 0, [neural(40)])
    for n in range(1, 21):
        assert foreground(step(recovery, image(n), n)) == []


def test_disappearance_during_pan_has_no_persistent_background_ghost(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    assert foreground(step(recovery, image(3, 47), 3))
    for n in range(4, 21):
        assert foreground(step(recovery, image(n), n)) == []


def test_translating_camera_invalidates_background_and_identity(tmp_path):
    rig, image = pan_fixture(tmp_path, translating=True)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    assert foreground(step(recovery, image(3, 47), 3)) == []


def test_pan_skips_frames_but_long_gap_and_reset_discard_identity(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    assert foreground(step(recovery, image(3, 47), 3))
    found = foreground(step(recovery, image(9, 65), 9))
    assert len(found) == 1
    assert np.allclose(found[0]['bbox'], [65, 95, 105, 125], atol=3)
    assert foreground(step(recovery, image(30, 68), 30, timestamp=2)) == []
    recovery.reset()
    assert step(recovery, image(0), 0) == []
    assert foreground(step(recovery, image(1, 40), 1)) == []


def test_pan_output_prefix_is_independent_of_future_images(tmp_path):
    rig, image = pan_fixture(tmp_path)
    first, second = ImageEvidence(rig), ImageEvidence(rig)
    prefixes = [[], []]
    for n in range(11):
        x = None if n == 0 else 40+3*(n-1)
        boxes = [neural(x)] if n in (1, 2) else []
        for i, recovery in enumerate((first, second)):
            prefixes[i].append(step(recovery, image(n, x), n, boxes))
    for n in range(11, 21):
        step(first, image(n, 70), n)
        step(second, image(n), n)
    assert prefixes[0] == prefixes[1]


def test_pan_measurement_does_not_expand_vehicle_to_dark_cast_shadow(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    current = image(3, 47)
    cv2.rectangle(current, (47, 125), (86, 145), (8, 8, 8), -1)
    found = foreground(step(recovery, current, 3))
    assert len(found) == 1
    assert np.allclose(found[0]['bbox'], [47, 95, 87, 125], atol=3)


def test_pan_rejects_attached_colour_incompatible_foreground(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    current = image(3, 47)
    cv2.rectangle(current, (87, 100), (120, 120), (15, 150, 25), -1)
    found = foreground(step(recovery, current, 3))
    assert len(found) == 1
    assert np.allclose(found[0]['bbox'], [47, 95, 87, 125], atol=3)


def test_pan_gradually_shrinking_fragment_cannot_replace_full_neural_anchor(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    for n in range(3, 15):
        width = round(40*.91**(n-2))
        current = image(n)
        cv2.rectangle(current, (47, 95), (47+width-1, 124), (195, 205, 215), -1)
        cv2.line(current, (50, 95), (50, 124), (80, 85, 90), 2)
        boxes = step(recovery, current, n)
        if width < 20:
            assert foreground(boxes) == []
            assert all(not box['measured'] or box['source'] == 'neural' for box in boxes)


def test_camera_motion_alone_cannot_renew_old_vehicle_identity(tmp_path):
    rig, image = pan_fixture(tmp_path)
    recovery = ImageEvidence(rig)
    step(recovery, image(0), 0)
    step(recovery, image(1, 40), 1, [neural(40)])
    step(recovery, image(2, 44), 2, [neural(44)])
    base = image(0, 47)
    corners = np.float32([[0, 0], [319, 0], [319, 179], [0, 179]])
    for n in range(3, 31):
        projected = []
        for pixel in corners:
            origin, ray = rig.ray(1, 0, pixel)
            projected.append(rig.project(1, n, origin+1_000_000*ray))
        transform = cv2.getPerspectiveTransform(corners, np.float32(projected))
        current = cv2.warpPerspective(base, transform, (320, 180), borderValue=(55, 55, 55))
        boxes = step(recovery, current, n)
        if n >= 23:
            assert foreground(boxes) == []


def test_quality_separates_internal_colour_holes_from_exterior_silhouette(evidence):
    seed(evidence)
    current = scene(48)
    cv2.rectangle(current, (62, 100), (82, 119), (30, 45, 60), -1)
    found = foreground(step(evidence, current, 2))
    assert len(found) == 1
    assert found[0]['quality']['foreground_fill'] < .8
    assert found[0]['quality']['silhouette_fill'] > .9
