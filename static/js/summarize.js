// Clear the file input when the trash icon is clicked
document.getElementById('clear-file').addEventListener('click', function() {
    document.getElementById('video-file').value = '';  // Clear file input value
    document.querySelector('.custom-file-label').innerHTML = 'Choose file';
});

// Update file input label with the chosen file name
document.getElementById('video-file').addEventListener('change', function() {
    let fileName = this.files[0]?.name || 'Choose file';
    document.querySelector('.custom-file-label').innerHTML = fileName;
});
