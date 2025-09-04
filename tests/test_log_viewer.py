"""
Test the scrollable log viewer functionality.
"""

from pathlib import Path
import sys
import tempfile

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from log_viewer import ScrollableLogViewer
from rich.text import Text


def test_log_viewer_basic_operations():
    """Test basic add and scroll operations."""
    viewer = ScrollableLogViewer(max_visible_lines=3, max_history=10)
    
    # Add some logs
    viewer.add_log("Log entry 1")
    viewer.add_log("Log entry 2")
    viewer.add_log("Log entry 3")
    viewer.add_log("Log entry 4")
    viewer.add_log("Log entry 5")
    
    # Check we have 5 logs
    assert len(viewer.logs) == 5
    
    # Check visible logs (should always be 3 due to padding)
    visible = viewer.get_visible_logs()
    assert len(visible) == 3
    # The last 3 actual logs should be visible (scrolled to bottom by default)
    assert visible[0].plain == "Log entry 3"
    assert visible[1].plain == "Log entry 4"
    assert visible[2].plain == "Log entry 5"
    
    # Test scrolling up
    viewer.scroll_up(2)
    visible = viewer.get_visible_logs()
    assert visible[0].plain == "Log entry 1"
    assert visible[1].plain == "Log entry 2"
    assert visible[2].plain == "Log entry 3"
    
    # Test scroll to bottom
    viewer.scroll_to_bottom()
    visible = viewer.get_visible_logs()
    assert visible[-1].plain == "Log entry 5"


def test_log_viewer_with_file_persistence(tmp_path: Path):
    """Test log persistence to file."""
    log_file = tmp_path / "test.log"
    viewer = ScrollableLogViewer(max_visible_lines=5, log_file=log_file)
    
    # Add logs
    viewer.add_log("First log entry")
    viewer.add_log("Second log entry", style="green")
    viewer.add_log(Text("Third log entry", style="red"))
    
    # Check file was created and contains logs
    assert log_file.exists()
    content = log_file.read_text()
    assert "First log entry" in content
    assert "Second log entry" in content
    assert "Third log entry" in content
    assert "Session started:" in content


def test_log_viewer_max_history():
    """Test that max_history limit is respected."""
    viewer = ScrollableLogViewer(max_visible_lines=2, max_history=5)
    
    # Add more logs than max_history
    for i in range(10):
        viewer.add_log(f"Log {i}")
    
    # Should only keep last 5
    assert len(viewer.logs) == 5
    assert viewer.logs[0].plain == "Log 5"
    assert viewer.logs[-1].plain == "Log 9"


def test_log_viewer_panel_generation():
    """Test panel generation with scroll indicators."""
    viewer = ScrollableLogViewer(max_visible_lines=2, max_history=10)
    
    # Add logs
    for i in range(5):
        viewer.add_log(f"Entry {i}")
    
    # Get panel at bottom
    panel = viewer.get_panel(title="Test Log")
    assert "[BOTTOM of 5]" in panel.title
    
    # Scroll to top
    viewer.scroll_to_top()
    panel = viewer.get_panel(title="Test Log")
    assert "[TOP of 5]" in panel.title
    
    # Scroll to middle
    viewer.scroll_down(1)
    panel = viewer.get_panel(title="Test Log")
    assert "of 5]" in panel.title