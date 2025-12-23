import pytest
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.api.app import clean_text

def test_clean_text_basic():
    text = "Hello World!"
    assert clean_text(text) == "hello world!"

def test_clean_text_special_chars():
    text = "Hello @World! #123"
    assert clean_text(text) == "hello world! 123"

def test_clean_text_whitespace():
    text = "  Hello   World  "
    assert clean_text(text) == "hello world"

def test_clean_text_empty():
    assert clean_text("") == ""

def test_clean_text_none():
    assert clean_text(None) == ""
