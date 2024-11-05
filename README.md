# Video Summarization and Download Application
This project is a web application for downloading YouTube videos or audio and creating summarized versions of them. It transcribes audio content, generates text summaries, maps key moments in the summary to video timestamps, and extracts relevant clips to compile a shorter video. The application supports only video summarization.

## Features
- **Video/Audio Download**: Download YouTube videos and audio using a custom downloader with `yt-dlp`.
- **Automatic Transcription**: Convert video/audio content to text using SpeechRecognition and Sphinx.
- **Text Summarization**: Generate summaries using `transformers` with Facebook's BART model.
- **Video Summarization**: Extract and concatenate key video moments based on the text summary.
- **Web Interface**: A simple Flask web app with Bootstrap for easy access to download and summarize functions.

## Tech Stack
- **Python Libraries**: `yt-dlp`, `moviepy`, `pydub`, `speech_recognition`, `transformers`, `spacy`, `sentence-transformers`, `sklearn`
- **Web Framework**: Flask with Flask-Bootstrap
- **Frontend**: HTML, CSS, JavaScript (with Bootstrap)

## Project Structure
```
.
├── main.py                     # Main Flask application.
├── download.py                 # Class for handling YouTube downloads.
├── transcript.py               # Class for transcription of video/audio files.
├── summarizer.py               # Class for generating video summaries.
├── static/
│   ├── css/
│   │   └── styles.css          # Custom CSS styles.
│   ├── images/
│   └── js/ ├── background.jpg  # Background image of Flask app.
│       │   └── icon.ico        # Image of Flask app icon.
│       └── summarize.js        # JavaScript for interaction in the summarize page.
├── templates/
│   ├── base.html               # Base template with navbar.
│   ├── download.html           # Page for downloading video/audio files.
│   └── summarize.html          # Page for summarizing video/audio content.
├── .env                        # Environment variables for paths and configurations.
├── requirements.txt            # Python dependencies.
├── LICENSE.md                  # License.
└── README.md                   # Project documentation.
```
## Installation
1. **Clone the repository:**
```
git clone https://github.com/fatihakb01/video-summarizer.git
cd video-summarizer
```
2. **Create a virtual environment and activate it:**
    ```bash
    # On Windows
    python -m venv venv
    venv\Scripts\activate

    # On MacOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```
3. **Install dependencies:**
```bash
pip install -r requirements.txt
```
4. **Set up environment variables. Create a .env file in the project root with the following:**
```
SECRET_KEY="your/secret/key" # example "dcuh436bcjds276xjfvirfd"
TRANSCRIPT_FOLDER=path/to/transcripts # example "files/transcripts/"
SUMMARIZE_FOLDER=path/to/summarized_videos # example "files/summarize/"
DOWNLOAD_FOLDER=path/to/downloads # example "files/downloads/"
```
5. **Download Spacy Model (for sentence segmentation):**
```bash
python -m spacy download en_core_web_md
```
## Usage
### Running the Application
1. **Start the Flask server:**
```bash
python main.py
```
2. **Open your browser and go to `http://127.0.0.1:5000`.**

### Application Pages
- **Download Page**: Enter a YouTube URL to download the video or audio.
- **Summarize Page**: Upload a video file to create a summarized version based on the content.

## Key Classes
### `Download`
- Handles downloading video or audio from YouTube using yt-dlp with multi-threaded downloading for improved speed.
- **Methods**:
    - `download`: Downloads video or audio, returning the file path if successful. 

### `Transcript`
- Extracts audio from video, splits it into chunks, transcribes each chunk in parallel, and saves the transcription.
- **Methods**:
    - `extract_audio_from_video`: Extracts audio from video files.
    - `split_audio`: Splits audio into chunks for parallel transcription.
    - `transcribe_in_parallel`: Transcribes chunks in parallel.
    - `save_transcript`: Saves the final transcript to a file.

### `Summarizer`
- Transcribes audio, summarizes the transcript, maps summary sentences to video timestamps, and extracts key video clips.
- **Methods**:
    - `summarize_transcript`: Summarizes the transcript in manageable chunks.
    - `map_summary_to_timestamps`: Maps summary sentences to video timestamps based on similarity scores.
    - `extract_important_clips`: Extracts video clips from specified timestamps.
    - `compile_clips`: Concatenates selected clips into a summarized video.

### `Cleaner`
- Utility class for clearing all files in a specified directory, ensuring the folder is emptied of previous files before new downloads or summaries.
- **Methods**:
    - `clean`: Deletes all files in the specified directory, logging any errors encountered during deletion.

### `Configuration`
- **Transcript Chunk Size**: You can adjust `chunk_length_ms` in `Transcript` to change the duration of audio chunks processed in parallel.
- **Target Summary Duration**: In `Summarizer`, set the `percentage` parameter in `summarize_video_pipeline` to control the length of the summarized video relative to the original.

## Future Improvements
- **Improve memory issue**: Summarization task will not work on all devices due to high memory requirement.
- **Alternative Summarization Models**: Experiment with different summarization models for varied summaries.
- **Enhanced UI**: Add more features in the UI for a better user experience.
- **Video Encoding Optimization**: Further optimize moviepy video handling to reduce memory usage and processing time.

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any new features or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.






