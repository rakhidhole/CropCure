function openFileSelector() {
    // Trigger a click on the hidden file input to open file selector
    document.getElementById("cropImage").click();
}

// Event listener for file input change
document.getElementById("cropImage").addEventListener("change", function() {
    if (this.files && this.files[0]) {
        // If a file is selected, submit the form
        document.getElementById("uploadMessage").textContent = "Uploading... Please wait!";
        
        const formData = new FormData();
        formData.append('image', this.files[0]); // Append the selected file to FormData

        // Use fetch to send the file to the server
        fetch('/analyze-image/', {  // Ensure this matches the URL defined in urls.py
            method: 'POST',
            body: formData,
            headers: {
                'X-CSRFToken': getCookie('csrftoken')  // Include CSRF token if using Django
            }
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            // Placeholder for actual disease detection process
            setTimeout(() => {
                document.getElementById("uploadMessage").innerHTML = `
                    <strong>Analysis Result:</strong>
                    <div class="result-box-container">
                        <div class="result-box">
                            <p class="result-title">Disease:</p>
                            <p>${data.disease_result}</p>
                        </div>
                        <div class="result-box">
                            <p class="result-title">Symptoms:</p>
                            <p>White spots on leaves, stunted growth</p>
                        </div>
                        <div class="result-box">
                            <p class="result-title">Treatment:</p>
                            <p>Apply fungicides, ensure good air circulation</p>
                        </div>
                    </div>
                `;
            }, 2000);
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById("uploadMessage").textContent = "Error uploading image.";
        });
    }
});

// Function to get CSRF token (if you don't have this function already)
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            // Check if this cookie string begins with the name we want
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}