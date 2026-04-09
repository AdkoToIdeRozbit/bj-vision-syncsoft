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


# ---------------------------------------------------------------------------
# detect_cut_card
# ---------------------------------------------------------------------------


def _make_white_frame(h: int = 10, w: int = 10) -> "np.ndarray":
    import numpy as np

    return np.full((h, w, 3), 255, dtype="uint8")


def _make_black_frame(h: int = 10, w: int = 10) -> "np.ndarray":
    import numpy as np

    return np.zeros((h, w, 3), dtype="uint8")


def test_detect_cut_card_all_white_returns_true():
    from app.vision.processor import detect_cut_card

    frame = _make_white_frame(20, 20)
    assert detect_cut_card(frame, (0, 0, 20, 20), threshold=0.3) is True


def test_detect_cut_card_all_black_returns_false():
    from app.vision.processor import detect_cut_card

    frame = _make_black_frame(20, 20)
    assert detect_cut_card(frame, (0, 0, 20, 20), threshold=0.3) is False


def test_detect_cut_card_mixed_below_threshold_returns_false():
    """Only a small corner is white; should fall below 0.3 threshold."""
    import numpy as np

    from app.vision.processor import detect_cut_card

    frame = np.zeros((20, 20, 3), dtype="uint8")
    # Make 2 x 2 = 4 pixels white out of 400 total → ratio 0.01
    frame[0:2, 0:2] = 255
    assert detect_cut_card(frame, (0, 0, 20, 20), threshold=0.3) is False


def test_detect_cut_card_mixed_above_threshold_returns_true():
    """More than 30% of the ROI is bright."""
    import numpy as np

    from app.vision.processor import detect_cut_card

    frame = np.zeros((10, 10, 3), dtype="uint8")
    # Make top 4 rows white → 40 of 100 pixels bright → ratio 0.40
    frame[0:4, :] = 255
    assert detect_cut_card(frame, (0, 0, 10, 10), threshold=0.3) is True


def test_detect_cut_card_empty_roi_returns_false():
    """A zero-size ROI must not crash and must return False."""
    from app.vision.processor import detect_cut_card

    frame = _make_white_frame(10, 10)
    # x1 == x2 produces zero-width region
    assert detect_cut_card(frame, (5, 5, 5, 10), threshold=0.3) is False


# ---------------------------------------------------------------------------
# LiveVideoProcessor — deck swap state machine
# ---------------------------------------------------------------------------


def _make_live_processor_mock_config(cut_card_roi=None):
    """Return a MagicMock VisionConfig that LiveVideoProcessor accepts."""
    import tempfile
    from unittest.mock import MagicMock

    import cv2
    import numpy as np

    from app.vision.processor import VisionConfig

    # LiveVideoProcessor reads the replay template from disk; create a tiny one
    tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
    img = np.full((10, 10), 128, dtype="uint8")
    cv2.imwrite(tmp.name, img)
    tmp.close()

    cfg = MagicMock(spec=VisionConfig)
    cfg.replay_template = tmp.name
    cfg.replay_roi = None
    cfg.replay_threshold = 0.5
    cfg.dealer_roi = None
    cfg.player_rois = {}
    cfg.card_threshold = 0.8
    cfg.lookback = 2
    cfg.cut_card_roi = cut_card_roi
    cfg.cut_card_threshold = 0.3
    return cfg, tmp.name


def _cleanup(path):
    import os

    try:
        os.unlink(path)
    except OSError:
        pass


def test_live_processor_no_cut_card_deck_num_stays_one():
    """Without a cut card ROI, all sessions carry deck_num=1."""
    from unittest.mock import patch

    import numpy as np

    from app.vision.processor import LiveVideoProcessor

    cfg, tmp = _make_live_processor_mock_config(cut_card_roi=None)
    processor = LiveVideoProcessor(cfg, [], [])

    # Simulate a replay event by patching the match and card detection
    with patch("app.vision.processor._detect_cards_for_session") as mock_detect:
        mock_detect.return_value = {
            "frame": 0,
            "dealer": ["ace"],
            "player_1": {},
            "player_2": {},
            "player_3": {},
            "player_4": {},
            "player_5": {},
            "player_6": {},
            "player_7": {},
        }
        with patch("cv2.matchTemplate") as mock_mt:
            # First frame: trigger replay
            mock_mt.return_value = np.full((1, 1), 0.9)
            frame = np.zeros((10, 10, 3), dtype="uint8")
            result = processor.process_frame(frame)

    assert result is not None
    assert result["deck_num"] == 1
    assert processor.deck_num == 1
    _cleanup(tmp)


def test_live_processor_cut_card_triggers_pending_swap():
    """When detect_cut_card returns True, pending_deck_swap is set."""
    from unittest.mock import patch

    import numpy as np

    from app.vision.processor import LiveVideoProcessor

    roi = (0, 0, 10, 10)
    cfg, tmp = _make_live_processor_mock_config(cut_card_roi=roi)
    processor = LiveVideoProcessor(cfg, [], [])

    with patch("app.vision.processor.detect_cut_card", return_value=True):
        with patch("cv2.matchTemplate") as mock_mt:
            mock_mt.return_value = np.full((1, 1), 0.0)  # no replay
            frame = np.zeros((10, 10, 3), dtype="uint8")
            processor.process_frame(frame)

    assert processor.pending_deck_swap is True
    assert processor.deck_num == 1  # not yet incremented — session hasn't ended
    _cleanup(tmp)


def test_live_processor_deck_increments_after_session_with_pending_swap():
    """deck_num increments after a session result is emitted when pending_deck_swap=True."""
    from unittest.mock import patch

    import numpy as np

    from app.vision.processor import LiveVideoProcessor

    roi = (0, 0, 10, 10)
    cfg, tmp = _make_live_processor_mock_config(cut_card_roi=roi)
    processor = LiveVideoProcessor(cfg, [], [])
    processor.pending_deck_swap = True  # pre-arm the swap

    with patch("app.vision.processor._detect_cards_for_session") as mock_detect:
        mock_detect.return_value = {
            "frame": 0,
            "dealer": ["king"],
            "player_1": {},
            "player_2": {},
            "player_3": {},
            "player_4": {},
            "player_5": {},
            "player_6": {},
            "player_7": {},
        }
        with patch("app.vision.processor.detect_cut_card", return_value=False):
            with patch("cv2.matchTemplate") as mock_mt:
                mock_mt.return_value = np.full((1, 1), 0.9)  # replay fires
                frame = np.zeros((10, 10, 3), dtype="uint8")
                result = processor.process_frame(frame)

    assert result is not None
    assert result["deck_num"] == 1  # this session was played on deck 1
    assert processor.deck_num == 2  # next session will be deck 2
    assert processor.pending_deck_swap is False
    _cleanup(tmp)


def test_live_processor_cut_card_not_checked_while_pending():
    """detect_cut_card must not be called when pending_deck_swap is True."""
    from unittest.mock import patch

    import numpy as np

    from app.vision.processor import LiveVideoProcessor

    roi = (0, 0, 10, 10)
    cfg, tmp = _make_live_processor_mock_config(cut_card_roi=roi)
    processor = LiveVideoProcessor(cfg, [], [])
    processor.pending_deck_swap = True

    with patch("app.vision.processor.detect_cut_card") as mock_dcc:
        with patch("cv2.matchTemplate") as mock_mt:
            mock_mt.return_value = np.full((1, 1), 0.0)
            processor.process_frame(np.zeros((10, 10, 3), dtype="uint8"))

    mock_dcc.assert_not_called()
    _cleanup(tmp)


def test_live_processor_consecutive_deck_swaps_undoes_first():
    """If two consecutive sessions both trigger a deck swap, the first swap is a
    false positive.  The processor should cancel it and stamp the second session
    with the pre-undo deck/session numbers before applying the real swap."""
    from unittest.mock import patch

    import numpy as np

    from app.vision.processor import LiveVideoProcessor

    roi = (0, 0, 10, 10)
    cfg, tmp = _make_live_processor_mock_config(cut_card_roi=roi)
    processor = LiveVideoProcessor(cfg, [], [])

    # Simulate state after a false first swap:
    #   deck 1 → (false swap) → deck 2, session_num reset to 1
    #   The last valid session was session 3 of deck 1.
    processor.deck_num = 2
    processor.session_num = 1
    processor.pending_deck_swap = True  # cut card already seen again
    processor._last_session_swapped = True  # first swap just fired
    processor._pre_swap_deck_num = 1  # deck before the false swap
    processor._pre_swap_session_num = (
        3  # session_num (post-increment) before the false swap reset
    )

    with patch("app.vision.processor._detect_cards_for_session") as mock_detect:
        mock_detect.return_value = {
            "frame": 0,
            "dealer": ["king"],
            "player_1": {},
            "player_2": {},
            "player_3": {},
            "player_4": {},
            "player_5": {},
            "player_6": {},
            "player_7": {},
        }
        with patch("app.vision.processor.detect_cut_card", return_value=False):
            with patch("cv2.matchTemplate") as mock_mt:
                mock_mt.return_value = np.full((1, 1), 0.9)  # replay fires
                frame = np.zeros((10, 10, 3), dtype="uint8")
                result = processor.process_frame(frame)

    # The false swap is undone: result should be stamped with deck 1, session 3
    assert result is not None
    assert result["deck_num"] == 1
    assert result["session"] == 3
    # The real swap is then applied: next session will be deck 2, session 1
    assert processor.deck_num == 2
    assert processor.session_num == 1
    assert processor._last_session_swapped is True
    _cleanup(tmp)
