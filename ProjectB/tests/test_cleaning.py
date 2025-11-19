from src.projectb.cleaning import clean_review


def test_clean_review_removes_html_and_whitespace():
    raw = "<p>This&nbsp;movie\n is awesome!</p>"
    cleaned = clean_review(raw)
    assert cleaned == "This movie is awesome!"
