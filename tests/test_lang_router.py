import pytest
from zesyde.lang_router import LanguageRouter, detect_language

from unittest.mock import MagicMock

def test_detect_language():
    assert detect_language('Hello world') == 'en'
    assert detect_language('你好') == 'zh'

def test_language_router():
    mock_router = MagicMock()
    # It returns a tuple of (router, routed_key, detected_lang, detected_sc)
    router = LanguageRouter(routers={'en': mock_router, 'zh': mock_router})
    result = router.route_for_text('Context with some english')
    assert result[0] == mock_router
    assert result[1] == 'en'

