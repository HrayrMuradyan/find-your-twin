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
    const resultsGrid = document.getElementById('results-grid');
    const loader = document.getElementById('loader');
    const resetControls = document.getElementById('reset-controls');
    const startAgainBtn = document.getElementById('start-again-btn');

    // --- State Variable ---
    let currentFile = null;

    // --- Uploader Event Listeners ---

    // Trigger file input when "Browse" button is clicked
    browseBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // Trigger file input when drop zone is clicked (only if no file is loaded)
    dropZone.addEventListener('click', (e) => {
        if (!currentFile && e.target !== browseBtn) {
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

    // --- SIMULATED Backend Search ---
    function uploadAndSearch(file) {
        console.log('Uploading file:', file.name);

        // Show loader and results section
        resultsSection.classList.remove('hidden');
        loader.classList.remove('hidden');
        resultsGrid.innerHTML = ''; // Clear previous results
        resetControls.classList.add('hidden'); // Hide start again button

        //
        // *** THIS IS THE SIMULATION ***
        //
        setTimeout(() => {
            // Hide the loader
            loader.classList.add('hidden');
            
            // Create mock results
            const mockResults = [
                { url: 'https://images.unsplash.com/photo-1520466809213-7b9a56c268af?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
                { url: 'https://images.unsplash.com/photo-1509967419530-da38b4704bc6?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
                { url: 'https://images.unsplash.com/photo-1542909168-82c3e726538d?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
                { url: 'https://images.unsplash.com/photo-1519085360753-af0119f7cbe7?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
                { url: 'https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
                { url: 'https://images.unsplash.com/photo-1531123897727-8f129e1688ce?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
                { url: 'https://images.unsplash.com/photo-1500648767791-00dcc994a43e?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
                { url: 'https://images.unsplash.com/photo-1506794778202-cad84cf45f1d?ixlib=rb-1.2.1&q=80&fm=jpg&crop=faces&fit=crop&h=200&w=200' },
            ];

            // Display the results
            displayResults(mockResults);

            // Show the "Start Again" button
            resetControls.classList.remove('hidden');

        }, 2000); // Simulate 2-second network delay
    }

    function displayResults(results) {
        resultsGrid.innerHTML = ''; // Clear again just in case
        results.forEach(result => {
            const img = document.createElement('img');
            img.src = result.url;
            img.alt = 'Similar Image Result';
            resultsGrid.appendChild(img);
        });
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
        resetControls.classList.add('hidden');
        resultsGrid.innerHTML = '';
        loader.classList.add('hidden');
    }

});