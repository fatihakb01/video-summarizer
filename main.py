# Import modules
import os
from flask import Flask, render_template, redirect, url_for, flash, request, send_file
from flask_bootstrap import Bootstrap5
from werkzeug.utils import secure_filename
from download import Download
from summarizer import Summarizer
from cleaner import Cleaner
from dotenv import load_dotenv

# Initialize the environment variables, Flask application, and Bootstrap extension
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['SUMMARIZE_FOLDER'] = os.getenv("SUMMARIZE_FOLDER")
app.config['DOWNLOAD_FOLDER'] = os.getenv("DOWNLOAD_FOLDER")
upload_cleaner = Cleaner(app.config['SUMMARIZE_FOLDER'])
download_cleaner = Cleaner(app.config['DOWNLOAD_FOLDER'])
Bootstrap5(app)

# Specify the video file extensions
video_extensions = {".mp4", ".avi", ".mov", ".mkv"}


# Define the homepage route for video/audio download
@app.route('/', methods=['GET', 'POST'])
def download_page():
    """
    Handles the video/audio download page. It accepts a YouTube URL and download type
    (video or audio) via a POST request, then downloads the file and serves it to the user.

    Returns:
    - On successful download, serves the file for download to the user.
    - On failure, flashes an error message and reloads the page.
    """
    # Ensure the download directory exists
    if not os.path.exists(app.config['DOWNLOAD_FOLDER']):
        os.makedirs(app.config['DOWNLOAD_FOLDER'])

    # Clean the download folder every time the browser is refreshed
    download_cleaner.clean()

    if request.method == 'POST':
        # Retrieve the URL and download type from the form submission
        url = request.form.get('url')
        download_type = request.form.get('download_type')

        if not url:
            flash("Please enter a valid YouTube URL.", "danger")
            return redirect(url_for('download_page'))

        # Create an instance of Download class
        downloader = Download(url)

        try:
            # Download either video or audio based on user's choice
            if download_type == 'video':
                file_path = downloader.download_video()
            elif download_type == 'audio':
                file_path = downloader.download_audio()
            else:
                flash("Please select a valid download type.", "danger")
                return redirect(url_for('download_page'))
        except Exception as e:
            flash(f"An error occurred while downloading the file: {e}", "danger")
            return redirect(url_for('download_page'))

        # Serve the downloaded file to the user
        if file_path:
            try:
                # Secure the file path and send it to the user for download
                filename = secure_filename(os.path.basename(file_path))
                sanitized_path = os.path.join(os.path.dirname(file_path), filename)
                return send_file(sanitized_path, as_attachment=True, download_name=filename)
            except Exception as e:
                flash(f"An error occurred while sending the file: {e}", "danger")
                return redirect(url_for('download_page'))

        flash("An error occurred. Please provide a valid Youtube url.", "danger")
        return redirect(url_for('download_page'))

    # Render the download page
    return render_template('download.html', info="")


# Define the route for video summarization functionality
@app.route('/summarize', methods=['GET', 'POST'])
def summarize_page():
    """
    Handles the video summarization page. Allows users to upload a video file,
    summarize it into a shorter version, and then download the summarized video.

    Returns:
    - If POST: Generates a summarized video file, then serves it for download.
    - If GET: Renders the summarization page with any relevant info.
    """
    # Ensure the upload directory exists
    if not os.path.exists(app.config['SUMMARIZE_FOLDER']):
        os.makedirs(app.config['SUMMARIZE_FOLDER'])

    #  Clean the download folder every time the browser is refreshed
    upload_cleaner.clean()

    if request.method == 'POST':
        # Check if a file was uploaded
        if 'video_file' not in request.files:
            return render_template('summarize.html', info="No file uploaded.")

        video_file = request.files['video_file']
        if video_file.filename == '':
            return render_template('summarize.html', info="No file selected.")

        # Check file extension to ensure it is a video file
        if video_file:
            filename = secure_filename(video_file.filename)
            file_ext = os.path.splitext(filename)[1].lower()

            # If an audio file is detected, show an error message and redirect
            if file_ext not in video_extensions:
                flash("Audio files are not supported for summarization. Please upload a video file.", "warning")
                return redirect(url_for('summarize_page'))

            # Secure the filename and prepare the file path
            file_path = os.path.join(app.config['SUMMARIZE_FOLDER'], filename)

            # Save the uploaded file to the server
            video_file.save(file_path)

            try:
                # Create an instance of Summarizer
                summarizer = Summarizer(file_path)
                # Generate a summarized version of the video
                summarized_video_path = summarizer.summarize_video_pipeline(0.2)

                if summarized_video_path:
                    # Serve the summarized video for download
                    response = send_file(summarized_video_path,
                                         as_attachment=True,
                                         download_name=f"summarized_{filename}")
                    return response
                else:
                    info = "Error: Summarization did not return a valid file path."
                    return render_template('summarize.html', info=info)
            # Display error if summarization fails
            except MemoryError as e:
                flash("Insufficient memory to complete summarization. Please try on a device with more memory.",
                      "error")
                info = f"Error during summarization: {e}"
                return render_template('summarize.html', info=info)
            except Exception as e:
                flash(f"An unexpected error occurred: {str(e)}", "error")
                info = f"Error during summarization: {e}"
                return render_template('summarize.html', info=info)

    # Render the summarization page
    return render_template('summarize.html', info="")


# Run the Flask application with debugging enabled and prevent duplicate execution
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
