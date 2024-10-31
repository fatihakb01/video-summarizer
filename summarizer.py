# Import necessary modules
import os
import spacy
import numpy as np
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transcript import Transcript


class Summarizer(Transcript):
    # Load models used for sentence embeddings and summarization once for efficiency
    sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_md")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def __init__(self, file, summarize_path="summarize", summary_name="summarize_video"):
        """
        Initializes the Summarizer class, setting paths and creating the summarization directory.

        Parameters:
        - file (str): Path to the video file.
        - summarize_path (str): Path to save summarized files.
        - summary_name (str): Name of the output summary video.
        """
        super().__init__(file)
        self.summarize_path = summarize_path
        self.summary_name = f"{self.summarize_path}/{summary_name}.mp4"

        # Ensure the summary directory exists, creating it if necessary
        if not os.path.exists(self.summarize_path):
            os.makedirs(self.summarize_path)

    def summarize_transcript(self, transcript):
        """
        Summarizes a transcript in chunks to fit within model constraints.

        Parameters:
        - transcript (str): Full transcript text to be summarized.

        Returns:
        - str: Concatenated summary of the transcript.
        """
        max_chunk_size = 1024
        transcript_chunks = [transcript[i:i + max_chunk_size] for i in range(0, len(transcript), max_chunk_size)]

        # Summarize each chunk and combine results
        summaries = [self.summarizer(chunk)[0]['summary_text'] for chunk in transcript_chunks]
        return f"Summary: {' '.join(summaries)}"

    def clear_summaries_folder(self):
        """
        Clears old files from the summary directory to prevent accumulation.
        """
        for file in os.listdir(self.summarize_path):
            file_path = os.path.join(self.summarize_path, file)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Could not delete file {file_path}: {e}")

    def split_transcript_into_sentences(self, transcript):
        """
        Splits the transcript into individual sentences.

        Parameters:
        - transcript (str): The full transcript text.

        Returns:
        - list[str]: List of sentences from the transcript.
        """
        doc = self.nlp(transcript)
        return [sent.text for sent in doc.sents]

    def get_sentence_embeddings(self, sentences):
        """
        Generates embeddings for sentences using a pre-trained sentence-transformers model.

        Parameters:
        - sentences (list[str]): List of sentences to be embedded.

        Returns:
        - np.ndarray: Array of sentence embeddings.
        """
        embeddings = self.sentence_embedder.encode(sentences, convert_to_numpy=True)
        return embeddings

    def get_similarity_scores(self, summary_sentences, transcript_sentences):
        """
        Calculates similarity scores between summary and transcript sentences.

        Parameters:
        - summary_sentences (list[str]): Summary sentences.
        - transcript_sentences (list[str]): Transcript sentences.

        Returns:
        - np.ndarray: Matrix of cosine similarity scores.
        """
        summary_embeddings = self.get_sentence_embeddings(summary_sentences)
        transcript_embeddings = self.get_sentence_embeddings(transcript_sentences)
        similarity_matrix = cosine_similarity(summary_embeddings, transcript_embeddings)
        return similarity_matrix

    def map_summary_to_timestamps(self, transcript, summary, video_length, top_n_clips):
        """
        Maps summary sentences to the transcript using similarity scores to determine key timestamps.

        Parameters:
        - transcript (str): Full transcript text.
        - summary (str): Summary of the transcript.
        - video_length (float): Duration of the original video in seconds.
        - top_n_clips (int): Number of important clips to select.

        Returns:
        - list[tuple]: List of selected (start, end) timestamps for key moments.
        """
        transcript_sentences = self.split_transcript_into_sentences(transcript)
        doc = self.nlp(summary)
        summary_sentences = [sent.text for sent in doc.sents]
        similarity_scores = self.get_similarity_scores(summary_sentences, transcript_sentences)

        used_indices = set()
        timestamps_with_scores = []

        for i, summary_sentence in enumerate(summary_sentences):
            most_similar_index = similarity_scores[i].argmax()

            if most_similar_index in used_indices:
                sorted_indices = np.argsort(-similarity_scores[i])
                for index in sorted_indices:
                    if index not in used_indices:
                        most_similar_index = index
                        break

            used_indices.add(most_similar_index)
            transcript_length = len(transcript_sentences)
            start_time = (most_similar_index / transcript_length) * video_length
            end_time = ((most_similar_index + 1) / transcript_length) * video_length
            timestamps_with_scores.append(((start_time, end_time), similarity_scores[i][most_similar_index]))

        timestamps_with_scores.sort(key=lambda x: x[1], reverse=True)
        selected_timestamps = [(start, end) for (start, end), score in timestamps_with_scores[:top_n_clips]]
        selected_timestamps.sort(key=lambda x: x[0])
        return selected_timestamps

    def extract_important_clips(self, timestamps):
        """
        Extracts key video clips based on given timestamps.

        Parameters:
        - timestamps (list[tuple]): List of (start, end) timestamps.

        Returns:
        - list[VideoFileClip]: List of video clips corresponding to the key moments.
        """
        video = VideoFileClip(self.name)
        important_clips = [video.subclip(start, end) for start, end in timestamps]
        return important_clips

    def compile_clips(self, clips):
        """
        Combines selected video clips into a single summarized video.

        Parameters:
        - clips (list[VideoFileClip]): List of video clips to be concatenated.
        """
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(self.summary_name, codec='libx264')

    def summarize_video_pipeline(self, percentage):
        """
        Full pipeline to transcribe, summarize, and compile the summarized video.

        Parameters:
        - percentage (float): Target percentage of the original video duration for the summary.

        Returns:
        - str: Path to the saved summarized video.
        """
        transcript = self.transcribe_video()
        if not transcript:
            print("Failed to transcribe the video.")
            return

        summary = self.summarize_transcript(transcript)
        video_length = VideoFileClip(self.name).duration
        target_duration = video_length * percentage

        temp_clips = self.map_summary_to_timestamps(transcript, summary, video_length, top_n_clips=1)
        if temp_clips:
            estimated_clip_duration = temp_clips[0][1] - temp_clips[0][0]
            top_n_clips = int(target_duration // estimated_clip_duration)
        else:
            print("Failed to map summary to timestamps.")
            return

        important_timestamps = self.map_summary_to_timestamps(transcript, summary, video_length, top_n_clips)
        clips = self.extract_important_clips(important_timestamps)
        self.compile_clips(clips)
        print(f"Summarized video saved as {self.summarize_path}")

        return self.summary_name
