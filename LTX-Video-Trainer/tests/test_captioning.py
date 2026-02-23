import sys
from unittest.mock import MagicMock

# Mock external dependencies to avoid installation overhead
sys.modules["torch"] = MagicMock()
sys.modules["diffusers"] = MagicMock()
sys.modules["transformers"] = MagicMock()

import pytest
# Now import the module under test
from ltxv_trainer.captioning import MediaCaptioningModel

class TestMediaCaptioningModel:
    def test_clean_raw_caption_standard_removal(self):
        """Test removal of standard prefix combinations."""
        # Test various combinations of start + kind + act
        assert MediaCaptioningModel._clean_raw_caption("The video shows a cat.") == "a cat."
        assert MediaCaptioningModel._clean_raw_caption("This image depicts a dog.") == "a dog."
        assert MediaCaptioningModel._clean_raw_caption("The scene features a bird.") == "a bird."
        assert MediaCaptioningModel._clean_raw_caption("This animated sequence presents a fish.") == "a fish."
        assert MediaCaptioningModel._clean_raw_caption("The video showcases a landscape.") == "a landscape."
        assert MediaCaptioningModel._clean_raw_caption("This image captures a moment.") == "a moment."

    def test_clean_raw_caption_no_change(self):
        """Test captions that should remain unchanged."""
        assert MediaCaptioningModel._clean_raw_caption("A cat running.") == "A cat running."
        assert MediaCaptioningModel._clean_raw_caption("Video of a dog.") == "Video of a dog."

        # "The video " matches start+kind, but "is" is not in act list
        assert MediaCaptioningModel._clean_raw_caption("The video is nice.") == "The video is nice."

        # Partial match of act
        assert MediaCaptioningModel._clean_raw_caption("The video sho a cat.") == "The video sho a cat."

    def test_clean_raw_caption_case_sensitivity(self):
        """Test that the cleaning is case-sensitive."""
        assert MediaCaptioningModel._clean_raw_caption("the video shows a cat.") == "the video shows a cat."
        assert MediaCaptioningModel._clean_raw_caption("THE VIDEO SHOWS A CAT.") == "THE VIDEO SHOWS A CAT."

    def test_clean_raw_caption_multiple_occurrences(self):
        """Test behavior with multiple occurrences of phrases.

        The implementation iterates through all combinations of (start, kind, act)
        and replaces the FIRST occurrence of each combination.
        """
        # Only the first occurrence of a specific phrase combination is removed
        # because the loop visits "The video shows" once and calls replace(..., 1)
        caption = "The video shows a cat. The video shows a dog."
        expected = "a cat. The video shows a dog."
        assert MediaCaptioningModel._clean_raw_caption(caption) == expected

        # Multiple different phrases are removed independently because they correspond
        # to different iterations of the loop.
        caption = "The video shows a cat. This image depicts a dog."
        expected = "a cat. a dog."
        assert MediaCaptioningModel._clean_raw_caption(caption) == expected

    def test_clean_raw_caption_edge_cases(self):
        """Test edge cases like empty strings and exact matches."""
        # Empty string
        assert MediaCaptioningModel._clean_raw_caption("") == ""

        # String consisting only of the prefix
        assert MediaCaptioningModel._clean_raw_caption("The video shows ") == ""

        # String with extra spaces after prefix
        assert MediaCaptioningModel._clean_raw_caption("The video shows  a cat.") == " a cat."
