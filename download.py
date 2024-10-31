# Import necessary modules
import os
import yt_dlp


class Download:
    """
    A class to handle downloading videos or audio from a specified URL and manage downloaded files.
    """

    def __init__(self, url, download_path="downloads"):
        """
        Initializes the Download object with the specified URL and download path.

        Parameters:
        - url (str): The URL of the video to download.
        - download_path (str): Directory to save downloaded files, default is 'downloads'.
        """
        self.url = url
        self.download_path = download_path
        self.name = None
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def clear_downloads_folder(self):
        """
        Clears the download directory by deleting all files in it.
        """
        for file in os.listdir(self.download_path):
            file_path = os.path.join(self.download_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
            except Exception as e:
                print(f"Could not delete file {file_path}: {e}")

    def download(self, filename, format_type):
        """
        Downloads video or audio from the URL in the specified format and saves it with a given filename.

        Parameters:
        - filename (str): Base filename to save the download as.
        - format_type (str): The format to download, e.g., 'best' for video or 'bestaudio' for audio.

        Returns:
        - str or None: The file path of the downloaded content if successful; None if an error occurs.
        """
        self.clear_downloads_folder()

        ydl_opts = {
            'format': format_type,
            'outtmpl': f'{self.download_path}/{filename}.%(ext)s',
            'quiet': True,
            'no_warnings': True,
            'external_downloader': 'aria2c',
            'external_downloader_args': ['-x', '16', '-k', '1M']
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            downloaded_files = [file for file in os.listdir(self.download_path) if filename in file]
            self.name = os.path.join(self.download_path, downloaded_files[0])
            print(f"Downloaded file to {self.name} successfully!")
            return self.name
        except Exception as e:
            print(f"An error occurred with yt-dlp: {e}")
            return None

    def download_audio(self):
        """
        Downloads audio only from the specified URL.

        Returns:
        - str or None: The file path of the downloaded audio if successful; None if an error occurs.
        """
        return self.download(filename="audio", format_type="bestaudio")

    def download_video(self):
        """
        Downloads video with audio from the specified URL.

        Returns:
        - str or None: The file path of the downloaded video if successful; None if an error occurs.
        """
        return self.download(filename="video", format_type="best")
