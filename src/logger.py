#!/usr/bin/env python3
"""
Universal Logger for NullBooth Scripts
Captures all stdout/stderr output and saves to timestamped log files.
"""

import sys
import os
from datetime import datetime
from pathlib import Path
from typing import TextIO, Optional


class TeeWriter:
    """A writer that forwards to multiple outputs (console + file)"""
    
    def __init__(self, *writers):
        self.writers = writers
    
    def write(self, text):
        for writer in self.writers:
            writer.write(text)
            writer.flush()
    
    def flush(self):
        for writer in self.writers:
            writer.flush()


class ScriptLogger:
    """
    Universal script logger that captures all output to both console and log files.
    
    Usage:
        logger = ScriptLogger("build_cov")
        logger.start_logging()
        # ... your script code ...
        logger.stop_logging()
    
    Or use as context manager:
        with ScriptLogger("build_cov"):
            # ... your script code ...
    """
    
    def __init__(self, script_name: str, log_dir: str = "logs"):
        self.script_name = script_name
        self.log_dir = Path(log_dir)
        self.log_file_path: Optional[Path] = None
        self.log_file: Optional[TextIO] = None
        self.original_stdout = None
        self.original_stderr = None
        
    def _create_log_file(self) -> Path:
        """Create timestamped log file"""
        # Ensure log directory exists
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{timestamp}_{self.script_name}.txt"
        log_file_path = self.log_dir / log_filename
        
        return log_file_path
    
    def start_logging(self):
        """Start capturing output to log file"""
        if self.log_file is not None:
            print("Warning: Logging is already started!")
            return
            
        # Create log file
        self.log_file_path = self._create_log_file()
        self.log_file = open(self.log_file_path, 'w', encoding='utf-8', buffering=1)
        
        # Store original streams
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Replace stdout and stderr with tee writers
        sys.stdout = TeeWriter(self.original_stdout, self.log_file)
        sys.stderr = TeeWriter(self.original_stderr, self.log_file)
        
        # Write header to log file
        header = f"""
{'='*80}
NullBooth Script Execution Log
{'='*80}
Script: {self.script_name}
Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log File: {self.log_file_path}
{'='*80}

"""
        print(header)
        
        print(f"🔍 Logging started for '{self.script_name}' script")
        print(f"📝 Output will be saved to: {self.log_file_path}")
        print("-" * 80)
    
    def stop_logging(self):
        """Stop capturing output and restore original streams"""
        if self.log_file is None:
            return
            
        # Write footer
        footer = f"""
{'-'*80}
Script execution completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Log saved to: {self.log_file_path}
{'='*80}
"""
        print(footer)
        
        # Restore original streams
        if self.original_stdout is not None:
            sys.stdout = self.original_stdout
            self.original_stdout = None
            
        if self.original_stderr is not None:
            sys.stderr = self.original_stderr
            self.original_stderr = None
            
        # Close log file
        if self.log_file is not None:
            self.log_file.close()
            self.log_file = None
            
        print(f"✅ Log saved to: {self.log_file_path}")
    
    def __enter__(self):
        """Context manager entry"""
        self.start_logging()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type is not None:
            print(f"\n❌ Script failed with exception: {exc_type.__name__}: {exc_val}")
        self.stop_logging()


# Convenience function
def setup_script_logging(script_name: str, log_dir: str = "logs") -> ScriptLogger:
    """
    Convenience function to setup script logging.
    
    Args:
        script_name: Name of the script (e.g., "build_cov", "train", "inference")
        log_dir: Directory to save log files (default: "logs")
    
    Returns:
        ScriptLogger instance
    
    Example:
        logger = setup_script_logging("build_cov")
        logger.start_logging()
        # ... script code ...
        logger.stop_logging()
    """
    return ScriptLogger(script_name, log_dir)


# Example usage as context manager
def log_script_execution(script_name: str, log_dir: str = "logs"):
    """
    Context manager for script logging.
    
    Usage:
        with log_script_execution("build_cov"):
            # ... your script code ...
    """
    return ScriptLogger(script_name, log_dir)


if __name__ == "__main__":
    # Test the logger
    with log_script_execution("test_script"):
        print("This is a test message")
        print("This will be saved to both console and log file")
        print("Testing error output", file=sys.stderr)
        print("Logger test completed")