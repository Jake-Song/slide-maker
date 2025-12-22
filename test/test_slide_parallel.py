from pathlib import Path
from unittest.mock import MagicMock, patch
from langchain.messages import AIMessage
from slide_parallel import synthesizer, GraphState

class TestAsyncImageOrdering:
    @patch('slide_parallel.save_image')
    @patch('slide_parallel.make_unique_dir')
    def test_async_generated_images_sorted_by_index(
        self,
        mock_make_dir: MagicMock,
        mock_save: MagicMock,
        tmp_path: Path
    ) -> None:
        """Verify that images generated asynchronously are sorted by index before saving.

        This test ensures that even when slides complete out of order (e.g., [2, 0, 1]),
        the synthesizer correctly sorts them by index before saving, resulting in the
        correct order [0, 1, 2].
        """
        mock_make_dir.return_value = tmp_path

        # Simulate slides arriving out of order from async workers
        state: GraphState = {
            "messages": [],
            "slides": [],
            "slides_with_index": [
                {"index": 2, "response": {"data": "slide2"}},
                {"index": 0, "response": {"data": "slide0"}},
                {"index": 1, "response": {"data": "slide1"}},
            ]
        }

        result = synthesizer(state)

        assert mock_make_dir.called
        assert mock_save.call_count == 3

        # Verify slides were saved in sorted order [0, 1, 2], not [2, 0, 1]
        call_indices = [call[0][2] for call in mock_save.call_args_list]
        assert call_indices == [0, 1, 2]

        assert "messages" in result
        assert isinstance(result["messages"][0], AIMessage)
        assert "successfully" in result["messages"][0].content.lower()
