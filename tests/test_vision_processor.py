"""Unit tests for app/vision/processor.py."""

from app.models import CardClass
from app.vision.processor import (
    _merge_detections,
    build_vision_config,
    map_card_names,
)

# ---------------------------------------------------------------------------
# map_card_names
# ---------------------------------------------------------------------------


def test_map_card_names_known_values():
    result = map_card_names(["ace", "ten", "king", "two"])
    assert result == [CardClass.ACE, CardClass.TEN, CardClass.KING, CardClass.TWO]


def test_map_card_names_unknown_falls_back():
    result = map_card_names(["ace", "joker", "bogus"])
    assert result[0] == CardClass.ACE
    assert result[1] == CardClass.UNKNOWN
    assert result[2] == CardClass.UNKNOWN


def test_map_card_names_empty():
    assert map_card_names([]) == []


def test_map_card_names_case_insensitive():
    result = map_card_names(["ACE", "TEN"])
    assert result == [CardClass.ACE, CardClass.TEN]


# ---------------------------------------------------------------------------
# build_vision_config
# ---------------------------------------------------------------------------


def test_build_vision_config_480p_sets_correct_paths():
    cfg = build_vision_config("video.mp4", "480p")
    assert cfg.video_path == "video.mp4"
    assert "480p" in cfg.replay_template
    assert "480p" in cfg.dealer_template_dir
    assert cfg.replay_threshold == 0.8
    assert cfg.card_threshold == 0.8


def test_build_vision_config_4k_sets_correct_paths():
    cfg = build_vision_config("video.mp4", "4k")
    assert "4k" in cfg.replay_template
    assert "4k" in cfg.dealer_template_dir


def test_build_vision_config_custom_thresholds():
    cfg = build_vision_config(
        "video.mp4", "480p", replay_threshold=0.9, card_threshold=0.75
    )
    assert cfg.replay_threshold == 0.9
    assert cfg.card_threshold == 0.75


def test_build_vision_config_480p_has_player_rois():
    cfg = build_vision_config("video.mp4", "480p")
    assert cfg.player_rois["player_4"] == (399, 290, 407, 374)
    assert cfg.player_rois["player_6"] == (525, 260, 533, 358)
    assert cfg.player_rois["player_1"] is None


def test_build_vision_config_4k_has_player_rois():
    cfg = build_vision_config("video.mp4", "4k")
    assert cfg.player_rois["player_1"] == (1045, 1183, 1077, 1467)
    assert cfg.player_rois["player_4"] == (1675, 1240, 1705, 1594)


# ---------------------------------------------------------------------------
# _merge_detections
# ---------------------------------------------------------------------------


def test_merge_detections_empty():
    assert _merge_detections([], 10, 10) == 0


def test_merge_detections_single_point():
    assert _merge_detections([(5, 5)], 10, 10) == 1


def test_merge_detections_overlapping_clusters_as_one():
    # Two very close points → same physical card
    points = [(5, 5), (6, 6)]
    assert _merge_detections(points, 20, 20) == 1


def test_merge_detections_separate_clusters():
    # Two far-apart points → distinct cards
    points = [(0, 0), (100, 100)]
    assert _merge_detections(points, 10, 10) == 2
