# Import modules
import os
import yt_dlp


class Download:
    def __init__(self, url, download_path="downloads"):
        self.url = url
        self.download_path = download_path
        self.name = None
        # Create the download directory if it doesn't exist
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def clear_downloads_folder(self):
        # Remove any existing files in the download path
        for file in os.listdir(self.download_path):
            file_path = os.path.join(self.download_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted old file: {file_path}")
            except Exception as e:
                print(f"Could not delete file {file_path}: {e}")

    def download(self, filename, format_type):
        # Clear the downloads folder before each new download
        self.clear_downloads_folder()

        # Set up yt-dlp options to download video directly to a file
        ydl_opts = {
            'format': format_type,  # Download the best video+audio or best available
            'outtmpl': f'{self.download_path}/{filename}.%(ext)s',  # Save file with correct extension
            'quiet': True,
            'no_warnings': True,
            'external_downloader': 'aria2c',  # Use aria2c for faster downloads
            'external_downloader_args': ['-x', '16', '-k', '1M']  # Example: 16 threads, 1MB chunk size
        }

        try:
            # Download the video directly to file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            # Return the downloaded file path
            downloaded_files = [file for file in os.listdir(self.download_path) if filename in file]
            self.name = os.path.join(self.download_path, downloaded_files[0])
            print(f"Downloaded file to {self.name} successfully!")
            return self.name
        except Exception as e:
            print(f"An error occurred with yt-dlp: {e}")
            return None

    def download_audio(self):
        return self.download(filename="audio", format_type="bestaudio")

    def download_video(self):
        return self.download(filename="video", format_type="best")

# # For testing purposes
# # Create video
# video_url = "https://www.youtube.com/watch?v=5HYPLcJ6XrM"
# download = Download(video_url)
# download.video()
# print(download.name)
