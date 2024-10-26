# Import modules.
import os
from flask import Flask, render_template, redirect, url_for, flash, request, send_file
from flask_bootstrap import Bootstrap5
from werkzeug.utils import secure_filename
from download import Download
from summarizer import Summarizer
from dotenv import load_dotenv

# Initialize the Flask application and Bootstrap extension.
load_dotenv()
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv("SECRET_KEY")
app.config['UPLOAD_FOLDER'] = os.getenv("UPLOAD_FOLDER")
Bootstrap5(app)


# Homepage for video/audio download
# In your download_page route
@app.route('/', methods=['GET', 'POST'])
def download_page():
    if request.method == 'POST':
        # Get the URL and download type from the form
        url = request.form.get('url')
        download_type = request.form.get('download_type')

        if not url:
            flash("Please enter a valid YouTube URL.", "danger")  # Flash error message
            return redirect(url_for('download_page'))

        # Create an instance of the Download class
        downloader = Download(url)

        try:
            # Handle download based on the user's selection
            if download_type == 'video':
                file_path = downloader.download_video()  # Download and merge video with audio
                # file_path = downloader.name  # Path to the merged video file
            elif download_type == 'audio':
                file_path = downloader.download_audio()  # Path to the downloaded audio file
            else:
                flash("Please select a valid download type.", "danger")
                return redirect(url_for('download_page'))
        except Exception as e:
            flash(f"An error occurred while downloading the file: {e}", "danger")
            return redirect(url_for('download_page'))

        # Serve the file for download
        if file_path:
            try:
                # Send the file and prompt the user to save it
                filename = secure_filename(os.path.basename(file_path))
                sanitized_path = os.path.join(os.path.dirname(file_path), filename)
                return send_file(sanitized_path, as_attachment=True, download_name=filename)
            except Exception as e:
                flash(f"An error occurred while sending the file: {e}", "danger")
                return redirect(url_for('download_page'))

        flash("An error occurred. File not found.", "danger")
        return redirect(url_for('download_page'))

    # If GET request, just render the page
    return render_template('download.html', info="")


# Summarize video page
@app.route('/summarize', methods=['GET', 'POST'])
def summarize_page():
    if request.method == 'POST':
        if 'video_file' not in request.files:
            return render_template('summarize.html', info="No file uploaded.")

        video_file = request.files['video_file']
        if video_file.filename == '':
            return render_template('summarize.html', info="No file selected.")

        if video_file:
            # Get a secure filename
            filename = secure_filename(video_file.filename)
            # Construct the full file path on the server (this could still be used for temporary storage)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

            # Ensure the upload folder exists
            if not os.path.exists(app.config['UPLOAD_FOLDER']):
                os.makedirs(app.config['UPLOAD_FOLDER'])

            # Save the file to the server
            video_file.save(file_path)

            try:
                # Create an instance of Summarizer with the uploaded file path
                summarizer = Summarizer(file_path)
                # Summarize the video and get the output file path
                summarized_video_path = summarizer.summarize_video_pipeline(top_n_clips=1)

                if summarized_video_path:
                    # Send the summarized video file for download
                    return send_file(summarized_video_path, as_attachment=True, download_name=f"summarized_{filename}")
                else:
                    info = "Error: Summarization did not return a valid file path."
                    return render_template('summarize.html', info=info)
            except Exception as e:
                info = f"Error during summarization: {e}"
                return render_template('summarize.html', info=info)

    return render_template('summarize.html', info="")


if __name__ == '__main__':
    app.run(debug=True)
