# Import necessary modules
import os
import spacy
import numpy as np
from moviepy.video.compositing.concatenate import concatenate_videoclips
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from transcript import Transcript


class Summarizer(Transcript):
    def __init__(self, url, summarize_path="summarize", summary_name="summarize_video"):
        super().__init__(url)
        self.nlp = spacy.load("en_core_web_md")
        self.transcript = self.transcribe_video()
        self.summarize_path = summarize_path
        self.summary_name = f"{summarize_path}/{summary_name}.mp4"
        # Create the transcript directory if it doesn't exist
        if not os.path.exists(self.summarize_path):
            os.makedirs(self.summarize_path)

    def summarize_transcript(self, summarization_model="facebook/bart-large-cnn"):
        """
        Use a transformer-based model to summarize the transcript.
        """
        # Load summarization model
        summarizer = pipeline("summarization", model=summarization_model)

        # Summarize transcript in chunks
        max_chunk_size = 1024
        transcript_chunks = [self.transcript[i:i + max_chunk_size]
                             for i in range(0, len(self.transcript), max_chunk_size)]

        summaries = [summarizer(chunk)[0]['summary_text'] for chunk in transcript_chunks]

        return f"Summary: {' '.join(summaries)}"

    def split_transcript_into_sentences(self):
        """
        Split the transcript into individual sentences using SpaCy.
        """
        doc = self.nlp(self.transcript)
        return [sent.text for sent in doc.sents]

    def get_sentence_embeddings(self, sentences):
        """
        Convert sentences into embeddings using SpaCy.
        """
        return np.array([self.nlp(sentence).vector for sentence in sentences])

    def get_similarity_scores(self, summary_sentences, transcript_sentences):
        """
        Compute cosine similarity between summary and transcript sentences.
        """
        summary_embeddings = self.get_sentence_embeddings(summary_sentences)
        transcript_embeddings = self.get_sentence_embeddings(transcript_sentences)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(summary_embeddings, transcript_embeddings)

        return similarity_matrix

    def map_summary_to_timestamps(self, video_length, summary):
        """
        Map summary sentences to transcript timestamps using sentence similarity.
        """
        # Step 1: Split transcript into sentences
        transcript_sentences = self.split_transcript_into_sentences()

        # Step 2: Split the summarized content into sentences
        doc = self.nlp(summary)  # Ensure you're using the summary here, not the transcript
        summary_sentences = [sent.text for sent in doc.sents]

        # Step 3: Compute similarity scores between summary and transcript sentences
        similarity_scores = self.get_similarity_scores(summary_sentences, transcript_sentences)

        # Step 4: For each summary sentence, find the most similar transcript sentence
        timestamps = []
        for i, summary_sentence in enumerate(summary_sentences):
            # Find the index of the most similar transcript sentence
            most_similar_index = similarity_scores[i].argmax()

            # Estimate the start and end time based on the position in the transcript
            transcript_length = len(transcript_sentences)
            start_time = (most_similar_index / transcript_length) * video_length
            end_time = ((most_similar_index + 1) / transcript_length) * video_length

            # Append the timestamp (start_time, end_time)
            timestamps.append((start_time, end_time))

        return timestamps

    def extract_important_clips(self, timestamps):
        """Extract important video clips based on the list of timestamps."""
        video = VideoFileClip(self.name)
        important_clips = [video.subclip(start, end) for start, end in timestamps]
        return important_clips

    def compile_clips(self, clips):
        """Combine important clips into one video."""
        final_clip = concatenate_videoclips(clips)
        final_clip.write_videofile(self.summary_name, codec='libx264')

    # Main pipeline
    def summarize_video_pipeline(self):
        # Step 1: Transcribe the video
        if not self.transcript:
            print("Failed to transcribe the video.")
            return

        # Step 2: Summarize the transcript
        summary = self.summarize_transcript()

        # Step 3: Map summary to timestamps
        video_length = VideoFileClip(self.name).duration
        important_timestamps = self.map_summary_to_timestamps(video_length, summary)

        # Step 4: Extract important video clips
        clips = self.extract_important_clips(important_timestamps)

        # Step 5: Compile the clips into a summarized video
        self.compile_clips(clips)
        print(f"Summarized video saved as {self.summarize_path}")


# # For testing purposes
# # Summarize transcript and create video based on summary
# video_url = "https://www.youtube.com/watch?v=5HYPLcJ6XrM"
# summarize = Summarizer(video_url)
# summ = summarize.summarize_transcript()
# print(summ)
# summarize.summarize_video_pipeline()
