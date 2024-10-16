# Import modules
from summarizer import Summarizer

# Summarize video
video_url = "https://www.youtube.com/watch?v=8UzgETz_UIM"
summarize = Summarizer(video_url)
summarize.merge()
summ = summarize.summarize_transcript()
print(summ)
summarize.summarize_video_pipeline()
