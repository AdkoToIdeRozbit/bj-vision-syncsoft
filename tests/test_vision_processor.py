"""Unit tests for app/vision/processor.py."""

from unittest.mock import patch

from app.models import CardClass
from app.vision.processor import (
    PlayerROIConfig,
    _merge_detections,
    build_vision_config,
    map_card_names,
    map_player_hands,
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
    assert cfg.player_rois["player_4"].default == (399, 290, 407, 374)  # type: ignore
    assert cfg.player_rois["player_6"].default == (525, 260, 533, 358)  # type: ignore
    assert cfg.player_rois["player_1"] is None


def test_build_vision_config_4k_has_player_rois():
    cfg = build_vision_config("video.mp4", "4k")
    assert cfg.player_rois["player_1"].default == (1045, 1183, 1077, 1467)  # type: ignore
    assert cfg.player_rois["player_4"].default == (1675, 1240, 1705, 1594)  # type: ignore


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


# ---------------------------------------------------------------------------
# map_player_hands
# ---------------------------------------------------------------------------


def test_map_player_hands_empty_returns_none():
    assert map_player_hands({}) is None


def test_map_player_hands_single_hand():
    result = map_player_hands({"hand1": ["ace", "king"]})
    assert result == {"hand1": [CardClass.ACE, CardClass.KING]}


def test_map_player_hands_split_hands():
    result = map_player_hands({"hand1": ["ace"], "hand2": ["queen", "three"]})
    assert result == {
        "hand1": [CardClass.ACE],
        "hand2": [CardClass.QUEEN, CardClass.THREE],
    }


def test_map_player_hands_unknown_card():
    result = map_player_hands({"hand1": ["ace", "joker"]})
    assert result == {"hand1": [CardClass.ACE, CardClass.UNKNOWN]}


# ---------------------------------------------------------------------------
# PlayerROIConfig split detection logic
# ---------------------------------------------------------------------------


def _make_frame():
    """Return a tiny black BGR frame for use in patched detection tests."""
    import numpy as np

    return np.zeros((10, 10, 3), dtype="uint8")


_DEFAULT_ROI = (0, 0, 5, 5)
_SPLIT1_ROI = (5, 0, 10, 5)
_SPLIT2_ROI = (0, 5, 5, 10)


def test_player_roi_default_cards_found_skips_splits():
    """When default ROI has cards, split ROIs must not be checked."""
    roi_cfg = PlayerROIConfig(
        default=_DEFAULT_ROI, split1=_SPLIT1_ROI, split2=_SPLIT2_ROI
    )
    call_log: list[tuple] = []

    def fake_detect(frame, roi, templates, threshold):
        call_log.append(roi)
        if roi == _DEFAULT_ROI:
            return ["ace"]
        return []

    with patch("app.vision.processor.detect_cards_in_roi", side_effect=fake_detect):
        from unittest.mock import MagicMock

        from app.vision.processor import _detect_cards_for_session

        config = MagicMock()
        config.dealer_roi = None
        config.player_rois = {"player_1": roi_cfg}
        config.card_threshold = 0.8

        frame = _make_frame()
        result = _detect_cards_for_session(0, frame, config, [], [])

    assert result is not None
    assert result["player_1"] == {"hand1": ["ace"]}
    assert _SPLIT1_ROI not in call_log
    assert _SPLIT2_ROI not in call_log


def test_player_roi_default_empty_tries_splits():
    """When default ROI is empty, split ROIs should be tried."""
    roi_cfg = PlayerROIConfig(
        default=_DEFAULT_ROI, split1=_SPLIT1_ROI, split2=_SPLIT2_ROI
    )

    def fake_detect(frame, roi, templates, threshold):
        if roi == _DEFAULT_ROI:
            return []
        if roi == _SPLIT1_ROI:
            return ["ten"]
        if roi == _SPLIT2_ROI:
            return ["queen"]
        return []

    with patch("app.vision.processor.detect_cards_in_roi", side_effect=fake_detect):
        from unittest.mock import MagicMock

        from app.vision.processor import _detect_cards_for_session

        config = MagicMock()
        config.dealer_roi = None
        config.player_rois = {"player_1": roi_cfg}
        config.card_threshold = 0.8

        result = _detect_cards_for_session(0, _make_frame(), config, [], [])

    assert result is not None
    assert result["player_1"] == {"hand1": ["ten"], "hand2": ["queen"]}


def test_player_roi_split1_empty_does_not_check_split2():
    """split2 should only be checked when split1 also has cards."""
    roi_cfg = PlayerROIConfig(
        default=_DEFAULT_ROI, split1=_SPLIT1_ROI, split2=_SPLIT2_ROI
    )
    call_log: list[tuple] = []

    def fake_detect(frame, roi, templates, threshold):
        call_log.append(roi)
        return []  # nothing found anywhere

    with patch("app.vision.processor.detect_cards_in_roi", side_effect=fake_detect):
        from unittest.mock import MagicMock

        from app.vision.processor import _detect_cards_for_session

        config = MagicMock()
        config.dealer_roi = None
        config.player_rois = {"player_1": roi_cfg}
        config.card_threshold = 0.8

        result = _detect_cards_for_session(0, _make_frame(), config, [], [])

    assert result is not None
    assert result["player_1"] == {}
    assert _SPLIT2_ROI not in call_log


def test_player_roi_no_split_rois_single_hand():
    """PlayerROIConfig with only default behaves like the old single-ROI path."""
    roi_cfg = PlayerROIConfig(default=_DEFAULT_ROI)

    def fake_detect(frame, roi, templates, threshold):
        return ["nine"]

    with patch("app.vision.processor.detect_cards_in_roi", side_effect=fake_detect):
        from unittest.mock import MagicMock

        from app.vision.processor import _detect_cards_for_session

        config = MagicMock()
        config.dealer_roi = None
        config.player_rois = {"player_1": roi_cfg}
        config.card_threshold = 0.8

        result = _detect_cards_for_session(0, _make_frame(), config, [], [])

    assert result is not None
    assert result["player_1"] == {"hand1": ["nine"]}
