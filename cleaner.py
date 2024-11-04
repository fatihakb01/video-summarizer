# Import module
import os


class Cleaner:
    """
    A utility class for clearing all files in a specified directory.
    """

    def __init__(self, path):
        """
        Initializes the Cleaner class with a target directory path.

        Parameters:
        - path (str): The path of the directory to clean.
        """
        self.path = path

    def clean(self):
        """
        Deletes all files in the specified directory.

        Iterates over all files in the directory specified by `self.path`
        and deletes each one.

        Raises:
        - Exception: If a file cannot be deleted, logs the exception error.
        """
        if self.path:
            for file in os.listdir(self.path):
                file_path = os.path.join(self.path, file)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted old file: {file_path}")
                except Exception as e:
                    print(f"Could not delete file {file_path}: {e}")
