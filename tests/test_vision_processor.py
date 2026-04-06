"""Unit tests for app/vision/processor.py."""

from unittest.mock import MagicMock, patch

from app.models import CardClass
from app.vision.processor import (
    LiveVideoProcessor,
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
# build_vision_config — deck swap defaults
# ---------------------------------------------------------------------------


def test_build_vision_config_deck_swap_defaults():
    cfg = build_vision_config("video.mp4", "480p")
    assert cfg.deck_swap_threshold_sec == 22.0
    assert cfg.fps == 30.0


def test_build_vision_config_deck_swap_custom():
    cfg = build_vision_config("video.mp4", "4k", deck_swap_threshold_sec=30.0, fps=15.0)
    assert cfg.deck_swap_threshold_sec == 30.0
    assert cfg.fps == 15.0


# ---------------------------------------------------------------------------
# LiveVideoProcessor — deck swap detection
# ---------------------------------------------------------------------------


def _make_processor(
    deck_swap_threshold_sec: float = 22.0,
    fps: float = 30.0,
) -> LiveVideoProcessor:
    """Build a LiveVideoProcessor with a mocked replay template."""
    cfg = build_vision_config(
        "dummy.mp4",
        "480p",
        deck_swap_threshold_sec=deck_swap_threshold_sec,
        fps=fps,
    )
    # Disable ROI slicing so the tiny test frame (10×10) is used directly
    cfg.replay_roi = None

    processor = LiveVideoProcessor.__new__(LiveVideoProcessor)
    processor.config = cfg
    processor.dealer_templates = []
    processor.player_templates = []
    processor.replay_template_img = MagicMock()
    processor.LOOKBACK = cfg.lookback
    from collections import deque

    processor.buffer = deque(maxlen=cfg.lookback + 1)
    processor.frame_index = 0
    processor.session_num = 1
    processor.replay_active = False
    processor.deck_number = 1
    processor._replay_last_disappeared = -1
    processor._deck_swap_declared = False
    processor._deck_swap_threshold_frames = int(deck_swap_threshold_sec * fps)
    return processor


def _drive_processor(
    processor: LiveVideoProcessor,
    replay_score: float,
    session_result: dict | None,
) -> dict | None:
    """Feed one frame into the processor with mocked cv2 calls."""
    import numpy as np

    frame = np.zeros((10, 10, 3), dtype="uint8")

    with (
        patch("cv2.matchTemplate"),
        patch("cv2.minMaxLoc", return_value=(0.0, replay_score, None, None)),
        patch(
            "app.vision.processor._detect_cards_for_session",
            return_value=session_result,
        ),
    ):
        return processor.process_frame(frame)


def test_no_deck_swap_within_normal_gap():
    """No deck-swap event when the gap is below the threshold."""
    processor = _make_processor(deck_swap_threshold_sec=22.0, fps=30.0)

    session_payload = {"frame": 0, "dealer": ["ace"], "player_1": {}}

    # First session
    _drive_processor(processor, 0.9, session_payload)
    # Replay disappears
    _drive_processor(processor, 0.5, None)
    # Advance only 200 frames (< 660 threshold)
    processor.frame_index += 200
    processor._replay_last_disappeared = processor.frame_index - 200

    # One more frame still within threshold — no deck_swap event
    result = _drive_processor(processor, 0.5, None)
    assert result is None
    assert processor.deck_number == 1


def test_deck_swap_event_fired_proactively():
    """A deck_swap event dict is returned the moment the threshold elapses."""
    processor = _make_processor(deck_swap_threshold_sec=22.0, fps=30.0)

    session_payload = {"frame": 0, "dealer": ["ace"], "player_1": {}}

    # First session
    result1 = _drive_processor(processor, 0.9, session_payload)
    assert result1 is not None
    assert result1["event"] == "session"
    assert result1["card_deck"] == 1

    # Replay disappears
    _drive_processor(processor, 0.5, None)

    # Jump past the threshold
    processor._replay_last_disappeared = 0
    processor.frame_index = 700

    # Next frame (no replay) should emit the deck_swap event
    swap_event = _drive_processor(processor, 0.5, None)
    assert swap_event is not None
    assert swap_event["event"] == "deck_swap"
    assert swap_event["card_deck"] == 2
    assert processor.deck_number == 2
    assert processor.session_num == 1

    # Subsequent frames must NOT re-fire the swap
    no_event = _drive_processor(processor, 0.5, None)
    assert no_event is None
    assert processor.deck_number == 2


def test_session_after_swap_uses_new_deck():
    """The first session detected after a swap carries the new deck number."""
    processor = _make_processor(deck_swap_threshold_sec=22.0, fps=30.0)

    session_payload = {"frame": 0, "dealer": ["ace"], "player_1": {}}

    _drive_processor(processor, 0.9, session_payload)
    _drive_processor(processor, 0.5, None)

    processor._replay_last_disappeared = 0
    processor.frame_index = 700

    # Fire the swap event
    swap = _drive_processor(processor, 0.5, None)
    assert swap["event"] == "deck_swap"
    assert swap["card_deck"] == 2

    # Now detect the first session of the new deck
    result = _drive_processor(processor, 0.9, session_payload)
    assert result is not None
    assert result["event"] == "session"
    assert result["card_deck"] == 2
    assert result["session"] == 1


def test_multiple_deck_swaps():
    """Multiple consecutive swaps each emit one deck_swap event."""
    processor = _make_processor(deck_swap_threshold_sec=22.0, fps=30.0)

    session_payload = {"frame": 0, "dealer": ["ace"], "player_1": {}}

    for expected_deck in range(1, 4):
        if expected_deck > 1:
            # Simulate a long gap to trigger the swap proactively
            processor._replay_last_disappeared = 0
            processor.frame_index = 700
            processor._deck_swap_declared = False

            swap = _drive_processor(processor, 0.5, None)
            assert swap is not None
            assert swap["event"] == "deck_swap"
            assert swap["card_deck"] == expected_deck

        result = _drive_processor(processor, 0.9, session_payload)
        assert result is not None
        assert result["event"] == "session"
        assert result["card_deck"] == expected_deck
        assert result["session"] == 1

        _drive_processor(processor, 0.5, None)


def test_no_swap_when_replay_never_appeared():
    """If the replay button has never been seen, no deck swap should fire."""
    processor = _make_processor(deck_swap_threshold_sec=22.0, fps=30.0)
    # _replay_last_disappeared starts at -1 (never seen)

    # Advance well past any threshold — still no swap because replay was never seen
    processor.frame_index = 1000

    result = _drive_processor(processor, 0.5, None)
    assert result is None
    assert processor.deck_number == 1


def test_deck_swap_threshold_frames_calculated_from_fps():
    """Threshold in frames scales correctly with different FPS values."""
    proc_30 = _make_processor(deck_swap_threshold_sec=22.0, fps=30.0)
    proc_15 = _make_processor(deck_swap_threshold_sec=22.0, fps=15.0)

    assert proc_30._deck_swap_threshold_frames == 660
    assert proc_15._deck_swap_threshold_frames == 330
