document.addEventListener('DOMContentLoaded', () => {

    // --- Tab Switching ---
    const navLinks = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-content');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = link.getAttribute('data-tab');

            // Update active link
            navLinks.forEach(nav => nav.classList.remove('active'));
            link.classList.add('active');

            // Update active content
            tabContents.forEach(content => {
                if (content.id === `${tab}-content`) {
                    content.classList.remove('hidden');
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                    content.classList.add('hidden');
                }
            });
        });
    });

    // --- File Uploader Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const browseBtn = document.getElementById('browse-btn');
    const uploadPreview = document.getElementById('upload-preview');
    const uploadControls = document.getElementById('upload-controls');
    const searchBtn = document.getElementById('search-btn');
    const removeBtn = document.getElementById('remove-btn');
    
    // --- Results Elements ---
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const topResultContainer = document.getElementById('top-result-container');
    const topResultImageWrapper = document.getElementById('top-result-image-wrapper');
    const matchPercentageNumber = document.getElementById('match-percentage-number');
    const resultsGrid = document.getElementById('results-grid'); // For other results
    const resetControls = document.getElementById('reset-controls');
    const startAgainBtn = document.getElementById('start-again-btn');

    // --- Image Modal Elements ---
    const imageModal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCloseBtn = document.querySelector('.modal-close-btn');

    // --- State Variable ---
    let currentFile = null;

    // --- Uploader Event Listeners ---

    // Trigger file input when "Browse" button is clicked
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); // Prevent dropZone's click event
        fileInput.click();
    });

    // Trigger file input when drop zone is clicked (only if no file is loaded)
    dropZone.addEventListener('click', (e) => {
        // Only allow click on dropZone itself, not child elements like buttons/preview
        if (!currentFile && e.target === dropZone) { 
            fileInput.click();
        }
    });

    // File input change event
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // Drag & Drop Events
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        if (!currentFile) {
            dropZone.classList.add('dragover');
        }
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        if (!currentFile) {
            dropZone.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file) {
                handleFile(file);
            }
        }
    });

    // --- File Handling ---
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }

        currentFile = file;

        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image Preview">`;
            dropZone.classList.add('file-loaded');
        };
        reader.readAsDataURL(file);

        // Show search/remove buttons
        uploadControls.classList.remove('hidden');
    }

    // --- Control Button Event Listeners ---

    // Search button
    searchBtn.addEventListener('click', () => {
        if (currentFile) {
            uploadControls.classList.add('hidden'); // Hide search/remove buttons
            uploadAndSearch(currentFile);
        }
    });

    // Remove button
    removeBtn.addEventListener('click', () => {
        resetApp();
    });

    // Start Again button
    startAgainBtn.addEventListener('click', () => {
        resetApp();
    });

    
    // *** --- THIS IS THE UPDATED FUNCTION --- ***
    async function uploadAndSearch(file) {
        console.log('Uploading file:', file.name);

        // Show loader and results section
        resultsSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        topResultContainer.classList.add('hidden'); // Hide top result elements
        resultsGrid.innerHTML = ''; // Clear previous results
        resetControls.classList.add('hidden'); // Hide start again button

        // Create a FormData object to send the file
        const formData = new FormData();
        formData.append("file", file);

        try {
            // Make the API call to your FastAPI backend
            const response = await fetch("/search/", {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            
            // Display the results from the API
            displayResults(data.results);

            // Show the "Start Again" button
            resetControls.classList.remove('hidden');

        } catch (error) {
            console.error("Error during search:", error);
            alert("Search failed. Could not connect to the backend. Please ensure it is running.");
        } finally {
            // Hide the loader whether the search succeeds or fails
            loader.classList.add('hidden');
        }
    }

    function displayResults(results) {
        resultsGrid.innerHTML = ''; // Clear again just in case

        // --- Handle Top Result ---
        if (results.length > 0) {
            const topResult = results[0];
            topResultContainer.classList.remove('hidden');
            topResultImageWrapper.innerHTML = `<img src="${topResult.url}" alt="Top Similar Image" data-fullsrc="${topResult.url}">`;
            
            // Animate and style the match percentage
            animateCount(matchPercentageNumber, topResult.similarity, 1500); 
            setMatchPercentageStyle(matchPercentageNumber, topResult.similarity);

            // Make the top result image clickable
            topResultImageWrapper.querySelector('img').addEventListener('click', () => {
                openModal(topResult.url);
            });
        }

        // --- Handle Other Results ---
        // We only want the next 5 (since we requested top 6 and [0] is the top result)
        const otherResults = results.slice(1, 6); 
        
        for (const result of otherResults) {
            const resultItem = document.createElement('div');
            resultItem.classList.add('result-item');

            const img = document.createElement('img');
            img.src = result.url;
            img.alt = 'Similar Image Result';
            img.dataset.fullsrc = result.url; // Store full size URL

            const matchSpan = document.createElement('span');
            matchSpan.classList.add('result-match-small');
            matchSpan.textContent = `${result.similarity}%`;
            setMatchPercentageStyle(matchSpan, result.similarity); 

            resultItem.appendChild(img);
            resultItem.appendChild(matchSpan);
            resultsGrid.appendChild(resultItem);

            // Make other result images clickable
            img.addEventListener('click', () => {
                openModal(result.url);
            });
        }
    }

    // --- Match Percentage Animation & Styling ---
    function animateCount(element, target, duration) {
        let start = 0;
        let increment = target / (duration / 16); // ~60fps
        let current = 0;
        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                element.textContent = target;
                clearInterval(timer);
            } else {
                element.textContent = Math.floor(current);
            }
        }, 16);
    }

    // Consolidated function for setting match percentage style
    function setMatchPercentageStyle(element, percentage) {
        // Remove existing color classes first
        element.classList.remove('match-green', 'match-orange', 'match-red');

        if (percentage > 80) {
            element.classList.add('match-green');
        } else if (percentage >= 50) {
            element.classList.add('match-orange');
        } else {
            element.classList.add('match-red');
        }
    }

    // --- Reset Function ---
    function resetApp() {
        // Reset state
        currentFile = null;
        fileInput.value = null; // Clear file input
        
        // Reset uploader UI
        uploadPreview.innerHTML = '';
        dropZone.classList.remove('file-loaded');
        uploadControls.classList.add('hidden');
        
        // Hide results UI
        resultsSection.classList.add('hidden');
        topResultContainer.classList.add('hidden');
        resultsGrid.innerHTML = '';
        resetControls.classList.add('hidden');
        loader.classList.add('hidden');
        
        // Reset match percentage display
        matchPercentageNumber.textContent = '0';
        matchPercentageNumber.classList.remove('match-green', 'match-orange', 'match-red');
    }

    // --- Image Modal Functionality ---
    function openModal(imageSrc) {
        modalImage.src = imageSrc;
        imageModal.classList.add('active');
        imageModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden'; // Prevent scrolling background
    }

    function closeModal() {
        imageModal.classList.add('hidden');
        imageModal.classList.remove('active');
        modalImage.src = ''; // Clear the image source
        document.body.style.overflow = ''; // Restore scrolling
    }

    // Close modal when clicking the close button
    modalCloseBtn.addEventListener('click', closeModal);

    // Close modal when clicking outside the image (on the overlay)
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal) {
            closeModal();
        }
    });

    // Close modal with Escape key
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && imageModal.classList.contains('active')) {
            closeModal();
        }
    });

});