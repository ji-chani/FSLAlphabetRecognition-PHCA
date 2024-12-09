import sys

class TeeOutput:
    def __init__(self, filename):
        self.terminal = sys.stdout  # Keep a reference to the original stdout
        self.log_file = open(filename, 'w')  # Open the log file for writing

    def write(self, message):
        self.terminal.write(message)  # Write to terminal
        self.log_file.write(message)  # Write to the log file

    def flush(self):
        # Ensure both the terminal and file are flushed
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        self.log_file.close()