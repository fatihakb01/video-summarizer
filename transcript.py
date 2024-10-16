# Import necessary modules
import os
import speech_recognition as sr
from moviepy.video.io.VideoFileClip import VideoFileClip
from download import Download


class Transcript(Download):
    def __init__(self, url, transcript_path="transcripts", transcript_name="video_transcript"):
        super().__init__(url)
        self.transcript_path = transcript_path
        self.transcript_name = f"{self.transcript_path}/{transcript_name}.txt"
        # Create the transcript directory if it doesn't exist
        if not os.path.exists(self.transcript_path):
            os.makedirs(self.transcript_path)

    def extract_audio_from_video(self, output_audio_file):
        """Extract the audio from a video file and save it as a WAV file."""
        try:
            if self.name is None:
                self.merge()
            video_clip = VideoFileClip(self.name)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(output_audio_file)
            print(f"Extracted audio saved to {output_audio_file}")
            return output_audio_file
        except Exception as e:
            print(f"An error occurred while extracting audio: {e}")
            return None

    def transcribe_video(self):
        """Transcribe the video using speech recognition."""
        # Extract the audio from the video and save as WAV file
        audio_file = "./downloads/temp_audio.wav"
        self.extract_audio_from_video(audio_file)

        # Initialize recognizer
        recognizer = sr.Recognizer()

        # Convert audio to text using Speech Recognition
        with sr.AudioFile(audio_file) as source:
            audio_data = recognizer.record(source)
            try:
                # Use speech recognition for transcription
                transcript = recognizer.recognize_sphinx(audio_data)
                print(f"Transcript: {transcript}")
                return transcript
            except sr.UnknownValueError:
                print("Speech Recognition could not understand the audio.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")

        # Clean up temporary audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)

        return None

    def save_transcript(self, transcript, filename="transcript.txt"):
        """Save the transcript to a text file."""
        file_path = os.path.join(self.transcript_path, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(transcript)
            print(f"Transcript saved to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving the transcript: {e}")


# # For testing purposes
# # Put transcript in a text file.
# video_url = "https://www.youtube.com/watch?v=5HYPLcJ6XrM"
# script = Transcript(video_url)
# audio_script = script.transcribe_video()
