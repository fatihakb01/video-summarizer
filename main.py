# Import modules.
import os
import yt_dlp
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip


def download_youtube_video(url, filename, format_type, download_path="downloads"):
    # Create the download directory if it doesn't exist
    if not os.path.exists(download_path):
        os.makedirs(download_path)

    # Set up yt-dlp options to download video directly to a file
    ydl_opts = {
        'format': format_type,  # Download the best video+audio or best available
        'outtmpl': f'{download_path}/{filename}.%(ext)s',  # Save file with correct extension
        'quiet': True,
        'no_warnings': True,
    }

    try:
        # Download the video directly to file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Return the downloaded file path
        downloaded_files = [file for file in os.listdir(download_path) if filename in file]
        file_path = os.path.join(download_path, downloaded_files[0])
        print(f"Downloaded file to {file_path} successfully!")
        return file_path
    except Exception as e:
        print(f"An error occurred with yt-dlp: {e}")
        return None


def merge_video_audio(video_file, audio_file, output_file):
    try:
        # Load video and audio files with moviepy
        video_clip = VideoFileClip(video_file)
        audio_clip = AudioFileClip(audio_file)

        # Set the audio of the video clip to the downloaded audio file
        video_with_audio = video_clip.set_audio(audio_clip)

        # Write the output to the desired file
        video_with_audio.write_videofile(output_file, codec='libx264', audio_codec='aac')

        print(f"Merged video saved to {output_file}")
    except Exception as e:
        print(f"An error occurred while merging video and audio: {e}")


# Download video and audio separately
video_url = "https://www.youtube.com/watch?v=zBjJUV-lzHo"
audio_path = download_youtube_video(url=video_url, filename="downloaded_audio", format_type="bestaudio")
video_path = download_youtube_video(url=video_url, filename="downloaded_video", format_type="bestvideo")

# Merge the downloaded audio and video files
if audio_path and video_path:
    output = "downloads/merged_output.mp4"
    merge_video_audio(video_path, audio_path, output)
