import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.app import get_topics

@pytest.fixture
def sample_data():
    data = {
        'review_text': [
            "This is a great product, I love it!",
            "This is a terrible product, I hate it!",
            "This is a good product, but it could be better.",
            "This is an amazing product, I would recommend it to anyone.",
            "This is a horrible product, I would not recommend it to anyone."
        ]
    }
    return pd.DataFrame(data)

def test_get_topics(sample_data):
    lda_model, vectorizer = get_topics(sample_data['review_text'])
    assert lda_model is not None
    assert vectorizer is not None
