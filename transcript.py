# Import modules
import os
import tempfile
import speech_recognition as sr
import mimetypes
import multiprocessing
from moviepy.video.io.VideoFileClip import VideoFileClip
from pydub import AudioSegment
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class Transcript:
    """
    A class for handling the transcription of video files by extracting audio,
    splitting it into manageable chunks, transcribing in parallel, and saving the result.
    """

    def __init__(self, file, transcript_path=os.getenv("TRANSCRIPT_FOLDER"), transcript_name="video_transcript",
                 chunk_length_ms=60000):
        """
        Initializes the Transcript object with input video file, output path, and transcript file name.

        Parameters:
        - file (str): Path to the input video file.
        - transcript_path (str): Directory for saving the transcript and temporary audio files.
        - transcript_name (str): Base name for the transcript file.
        - chunk_length_ms (int): Duration of each audio chunk in milliseconds for parallel processing.
        """
        self.name = file
        self.transcript_path = transcript_path
        self.transcript_name = f"{self.transcript_path}/{transcript_name}.txt"
        self.audio_file = f"{self.transcript_path}/temp_audio.wav"
        self.chunk_length_ms = chunk_length_ms
        self.recognizer = sr.Recognizer()

        # Ensure the transcript directory exists
        if not os.path.exists(self.transcript_path):
            os.makedirs(self.transcript_path)

    def is_audio_file(self):
        """
        Checks if the input file is an audio file based on its MIME type.

        Returns:
        - bool: True if the file is audio, False otherwise.
        """
        mime_type, _ = mimetypes.guess_type(self.name)
        return mime_type and mime_type.startswith('audio')

    def extract_audio_from_video(self):
        """
        Extracts audio from the video file and saves it as a separate audio file.

        Returns:
        - str or None: Path of the extracted audio file if successful; None if an error occurs.
        """
        try:
            video_clip = VideoFileClip(self.name)
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(self.audio_file)
            return self.audio_file
        except Exception as e:
            print(f"An error occurred while extracting audio: {e}")
            return None

    def split_audio(self):
        """
        Splits the audio file into smaller chunks for parallel processing.

        Returns:
        - list of str: List of file paths for each audio chunk created.
        """
        audio = AudioSegment.from_file(self.audio_file)
        chunk_files = []

        for i in range(0, len(audio), self.chunk_length_ms):
            chunk = audio[i:i + self.chunk_length_ms]
            temp_file_path = tempfile.mktemp(suffix=".wav")
            chunk.export(temp_file_path, format="wav")
            chunk_files.append(temp_file_path)

        return chunk_files

    def transcribe_chunk(self, chunk_file):
        """
        Transcribes a single audio chunk using the Sphinx recognizer.

        Parameters:
        - chunk_file (str): Path to the audio chunk file.

        Returns:
        - str: Transcribed text from the audio chunk or an empty string if an error occurs.
        """
        with sr.AudioFile(chunk_file) as source:
            audio_data = self.recognizer.record(source)
            try:
                return self.recognizer.recognize_sphinx(audio_data)
            except sr.UnknownValueError:
                return ""
            except sr.RequestError as e:
                print(f"Request error during transcription: {e}")
                return ""

    def transcribe_in_parallel(self):
        """
        Transcribes all audio chunks in parallel to improve performance.

        Returns:
        - str: Combined transcription text from all audio chunks.
        """
        audio_chunk_files = self.split_audio()
        with multiprocessing.Pool() as pool:
            transcripts = pool.map(self.transcribe_chunk, audio_chunk_files)

        for chunk in audio_chunk_files:
            try:
                os.remove(chunk)
            except Exception as e:
                print(f"Error deleting temporary file {chunk}: {e}")
        return " ".join(transcripts)

    def transcribe_video(self):
        """
        Manages the full transcription process for video files.

        Returns:
        - str: Final combined transcription text.
        """
        self.extract_audio_from_video()
        transcript = self.transcribe_in_parallel()

        if os.path.exists(self.audio_file):
            os.remove(self.audio_file)

        return transcript

    def save_transcript(self, transcript, filename="transcript.txt"):
        """
        Saves the final transcript text to a specified file.

        Parameters:
        - transcript (str): The transcribed text to be saved.
        - filename (str): Name of the file to save the transcript in (default is "transcript.txt").
        """
        file_path = os.path.join(self.transcript_path, filename)
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(transcript)
            print(f"Transcript saved to {file_path}")
        except Exception as e:
            print(f"An error occurred while saving the transcript: {e}")
