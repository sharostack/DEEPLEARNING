"""Small validation helpers shared by DensePose inference and synthetic tests."""


def first_densepose_detection(predictions):
    """Return the first extracted DensePose result or raise on no/invalid result."""
    if predictions is None or len(predictions) == 0:
        raise RuntimeError("DensePose produced no person detections; sample extraction failed")
    try:
        result, box = predictions[0]
    except (TypeError, ValueError, IndexError) as exc:
        raise RuntimeError("DensePose extractor returned an invalid detection structure") from exc
    if result is None or box is None:
        raise RuntimeError("DensePose extractor returned an empty result or box")
    return result, box
