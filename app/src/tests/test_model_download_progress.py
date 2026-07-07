from voice2text.stt.model_download import format_download_progress


def test_download_progress_with_known_total_uses_percent_and_total_mb():
    message = format_download_progress(
        "download",
        "whisperx:align-pt",
        50 * 1024 * 1024,
        100 * 1024 * 1024,
    )

    assert "[download] download downloading: whisperx:align-pt" in message
    assert "50%" in message
    assert "(50.0/100.0 MB)" in message


def test_download_progress_with_unknown_total_is_bytes_only():
    message = format_download_progress(
        "download",
        "whisperx:align-pt",
        64 * 1024 * 1024,
        None,
    )

    assert message == (
        "[download] download downloading: whisperx:align-pt "
        "(64.0 MB downloaded; total unknown)"
    )
    assert ">" not in message
    assert "%" not in message
