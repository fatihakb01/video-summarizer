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
    sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')
    nlp = spacy.load("en_core_web_md")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

    def __init__(self, file, summarize_path="summarize", summary_name="summarize_video"):
        super().__init__(file)
        # Load sentence-transformers for contextual sentence embeddings
        # self.sentence_embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # self.nlp = spacy.load("en_core_web_md")
        # self.transcript = self.transcribe_video()
        self.summarize_path = summarize_path
        self.summary_name = f"{summarize_path}/{summary_name}.mp4"

        # Create the transcript directory if it doesn't exist
        if not os.path.exists(self.summarize_path):
            os.makedirs(self.summarize_path)

    def summarize_transcript(self, transcript):
        """
        Use a transformer-based model to summarize the transcript.
        """

        # Summarize transcript in chunks
        max_chunk_size = 1024
        transcript_chunks = [transcript[i:i + max_chunk_size]
                             for i in range(0, len(transcript), max_chunk_size)]

        summaries = [self.summarizer(chunk)[0]['summary_text'] for chunk in transcript_chunks]

        return f"Summary: {' '.join(summaries)}"

    def split_transcript_into_sentences(self, transcript):
        """
        Split the transcript into individual sentences using SpaCy.
        """
        doc = self.nlp(transcript)
        return [sent.text for sent in doc.sents]

    def get_sentence_embeddings(self, sentences):
        """
        Convert sentences into embeddings using SentenceTransformer for contextual embeddings.
        """
        # Use the sentence-transformers model to get contextual embeddings
        embeddings = self.sentence_embedder.encode(sentences, convert_to_numpy=True)
        return embeddings

    def get_similarity_scores(self, summary_sentences, transcript_sentences):
        """
        Compute cosine similarity between summary and transcript sentences using contextual embeddings.
        """
        summary_embeddings = self.get_sentence_embeddings(summary_sentences)
        transcript_embeddings = self.get_sentence_embeddings(transcript_sentences)

        # Compute pairwise cosine similarity
        similarity_matrix = cosine_similarity(summary_embeddings, transcript_embeddings)

        return similarity_matrix

    def map_summary_to_timestamps(self, transcript, summary, video_length, top_n_clips):
        """
        Map summary sentences to transcript timestamps using sentence similarity.
        Select the top N most important clips based on similarity scores.
        """
        # Step 1: Split transcript into sentences
        transcript_sentences = self.split_transcript_into_sentences(transcript)

        # Step 2: Split the summarized content into sentences
        doc = self.nlp(summary)
        summary_sentences = [sent.text for sent in doc.sents]

        # Step 3: Compute similarity scores between summary and transcript sentences
        similarity_scores = self.get_similarity_scores(summary_sentences, transcript_sentences)

        # Step 4: For each summary sentence, find the most similar transcript sentence
        used_indices = set()
        timestamps_with_scores = []

        for i, summary_sentence in enumerate(summary_sentences):
            most_similar_index = similarity_scores[i].argmax()

            if most_similar_index in used_indices:
                # Find the next most similar sentence that hasn't been used yet
                sorted_indices = np.argsort(-similarity_scores[i])
                for index in sorted_indices:
                    if index not in used_indices:
                        most_similar_index = index
                        break

            used_indices.add(most_similar_index)

            transcript_length = len(transcript_sentences)
            start_time = (most_similar_index / transcript_length) * video_length
            end_time = ((most_similar_index + 1) / transcript_length) * video_length

            # Append the timestamp along with the similarity score for ranking later
            timestamps_with_scores.append(((start_time, end_time), similarity_scores[i][most_similar_index]))

        # Step 5: Sort timestamps by their similarity scores (descending order)
        timestamps_with_scores.sort(key=lambda x: x[1], reverse=True)

        # Step 6: Select the top N most important clips based on similarity scores
        selected_timestamps = [(start, end) for (start, end), score in timestamps_with_scores[:top_n_clips]]

        # Ensure the timestamps are in chronological order
        selected_timestamps.sort(key=lambda x: x[0])

        return selected_timestamps

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
    def summarize_video_pipeline(self, top_n_clips):
        # Step 1: Transcribe the video
        transcript = self.transcribe_video()
        if not transcript:
            print("Failed to transcribe the video.")
            return

        # Step 2: Summarize the transcript
        summary = self.summarize_transcript(transcript)

        # Step 3: Map summary to timestamps
        video_length = VideoFileClip(self.name).duration
        important_timestamps = self.map_summary_to_timestamps(transcript, summary, video_length, top_n_clips)

        # Step 4: Extract important video clips
        clips = self.extract_important_clips(important_timestamps)

        # Step 5: Compile the clips into a summarized video
        self.compile_clips(clips)
        print(f"Summarized video saved as {self.summarize_path}")

        return self.summary_name

# # For testing purposes
# # Summarize transcript and create video based on summary
# video_url = "https://www.youtube.com/watch?v=5HYPLcJ6XrM"
# summarize = Summarizer(video_url)
# summ = summarize.summarize_transcript()
# print(summ)
# summarize.summarize_video_pipeline()
