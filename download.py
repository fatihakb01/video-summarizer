# Import modules
import os
import yt_dlp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


class Download:
    def __init__(self, url, download_path="downloads"):
        self.url = url
        self.download_path = download_path
        self.name = None
        # Create the download directory if it doesn't exist
        if not os.path.exists(self.download_path):
            os.makedirs(self.download_path)

    def download(self, filename, format_type, name="video"):
        # Define the output file name
        self.name = f"{self.download_path}/{name}.mp4"

        # Set up yt-dlp options to download video directly to a file
        ydl_opts = {
            'format': format_type,  # Download the best video+audio or best available
            'outtmpl': f'{self.download_path}/{filename}.%(ext)s',  # Save file with correct extension
            'quiet': True,
            'no_warnings': True
        }

        try:
            # Download the video directly to file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.url])

            # Return the downloaded file path
            downloaded_files = [file for file in os.listdir(self.download_path) if filename in file]
            file_path = os.path.join(self.download_path, downloaded_files[0])
            print(f"Downloaded file to {file_path} successfully!")
            return file_path
        except Exception as e:
            print(f"An error occurred with yt-dlp: {e}")
            return None

    def download_audio(self):
        return self.download(filename="downloaded_audio", format_type="bestaudio")

    def download_video(self):
        return self.download(filename="downloaded_video", format_type="bestvideo")

    def merge(self):
        try:
            # Load video and audio files with moviepy
            video_clip = VideoFileClip(self.download_video())
            audio_clip = AudioFileClip(self.download_audio())

            # Set the audio of the video clip to the downloaded audio file
            video_with_audio = video_clip.set_audio(audio_clip)

            # Write the output to the desired file
            video_with_audio.write_videofile(self.name, codec='libx264', audio_codec='aac')
            print(f"Merged video saved to {self.name}")
        except Exception as e:
            print(f"An error occurred while merging video and audio: {e}")


# # For testing purposes
# # Create video
# video_url = "https://www.youtube.com/watch?v=5HYPLcJ6XrM"
# download = Download(video_url)
# download.merge()
# print(download.name)
