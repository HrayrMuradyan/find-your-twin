document.addEventListener('DOMContentLoaded', () => {

    // --- CONFIGURATION ---
    
    // 1. Define your production URL (Hugging Face)
    const PROD_API_URL = "https://hrayrmuradyan-find-your-twin.hf.space";
    
    // 2. Define your local URL (Localhost Python)
    const LOCAL_API_URL = "http://127.0.0.1:8000";

    // 3. Automatically select based on where the browser is running
    // If the domain is localhost, 127.0.0.1, or if you opened the file directly (file://)
    const isLocalEnvironment = 
        window.location.hostname === 'localhost' || 
        window.location.hostname === '127.0.0.1' || 
        window.location.protocol === 'file:';

    const API_BASE_URL = isLocalEnvironment ? LOCAL_API_URL : PROD_API_URL;

    console.log(`Environment detected: ${isLocalEnvironment ? 'DEVELOPMENT' : 'PRODUCTION'}`);
    console.log(`Using API: ${API_BASE_URL}`);

    // --- Tab Switching ---
    const navLinks = document.querySelectorAll('.nav-link');
    const tabContents = document.querySelectorAll('.tab-content');

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const tab = link.getAttribute('data-tab');
            navLinks.forEach(nav => nav.classList.remove('active'));
            link.classList.add('active');
            tabContents.forEach(content => {
                if (content.id === `${tab}-content`) {
                    content.classList.remove('hidden');
                    content.classList.add('active');
                } else {
                    content.classList.remove('active');
                    content.classList.add('hidden');
                }
            });
            window.scrollTo({ top: 0, behavior: 'auto' }); 
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
    const consentCheckbox = document.getElementById('consent-checkbox');
    const consentBox = document.querySelector('.consent-box');
    const searchErrorMessage = document.getElementById('search-error-message');
    const errorText = document.getElementById('error-text');
    
    // --- Results Elements ---
    const resultsSection = document.getElementById('results-section');
    const loader = document.getElementById('loader');
    const topResultContainer = document.getElementById('top-result-container');
    const topResultImageWrapper = document.getElementById('top-result-image-wrapper');
    const matchPercentageNumber = document.getElementById('match-percentage-number');
    const otherResultsTitle = document.querySelector('.other-results-title');
    const resetControls = document.getElementById('reset-controls');
    const startAgainBtn = document.getElementById('start-again-btn');
    const saveInfo = document.getElementById('save-info');
    const userUuid = document.getElementById('user-uuid');
    const copyUuidBtn = document.getElementById('copy-uuid-btn'); 

    // --- Carousel Elements ---
    const carouselContainer = document.getElementById('carousel-container');
    const carouselTrack = document.getElementById('carousel-track');
    const carouselPrev = document.getElementById('carousel-prev');
    const carouselNext = document.getElementById('carousel-next');

    // --- Carousel State ---
    let currentIndex = 0; 
    let itemsPerPage = 5;
    let totalItems = 0;

    // --- Image Modal Elements ---
    const imageModal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCloseBtn = document.querySelector('.modal-close-btn');

    // --- About Page CTA ---
    const ctaTryNowBtn = document.getElementById('cta-try-now');
    const consentPrivacyLink = document.getElementById('consent-privacy-link'); 
    const aboutPrivacyLink = document.querySelector('.about-privacy-link'); 

    // --- NEW: Manage Data Elements ---
    const deleteForm = document.getElementById('delete-form');
    const uuidInput = document.getElementById('uuid-input');
    const deleteBtn = document.getElementById('delete-btn');
    const deleteMessage = document.getElementById('delete-message');

    // --- State Variable ---
    let currentFile = null;
    
    // --- Responsive Carousel ---
    function updateItemsPerPage() {
        if (window.innerWidth < 600) itemsPerPage = 2;
        else if (window.innerWidth < 900) itemsPerPage = 3;
        else if (window.innerWidth < 1100) itemsPerPage = 4;
        else itemsPerPage = 5;
    }
    updateItemsPerPage(); 

    window.addEventListener('resize', () => {
        updateItemsPerPage();
        if (totalItems > 0) {
            updateCarouselPosition(); 
        }
    });


    // --- Uploader Event Listeners ---
    
    // 1. Browse Button
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); 
        fileInput.click();
    });

    // This allows clicking to replace the image even if one is already uploaded
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('#upload-instructions') || e.target === dropZone || e.target.closest('#upload-preview')) { 
            fileInput.click();
        }
    });

    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) {
            handleFile(file);
        }
        // Reset value to allow selecting the same file again if needed
        e.target.value = '';
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        const file = e.dataTransfer.files[0];
        if (file) {
            handleFile(file);
        }
    });

    // --- File Handling ---
    function handleFile(file) {
        if (!file.type.startsWith('image/')) {
            alert('Please upload an image file.');
            return;
        }
        currentFile = file;

        searchErrorMessage.classList.add('hidden');
        resultsSection.classList.add('hidden');     
        resetControls.classList.add('hidden');     

        const reader = new FileReader();
        reader.onload = (e) => {
            uploadPreview.innerHTML = `<img src="${e.target.result}" alt="Uploaded Image Preview">`;
            dropZone.classList.add('file-loaded');
        };
        reader.readAsDataURL(file);
        
        uploadControls.classList.remove('hidden');
        
        setTimeout(() => {
            uploadControls.scrollIntoView({ behavior: 'smooth', block: 'center' });
            setTimeout(() => {
                consentBox.classList.add('highlight-attention');
            }, 500); 
            setTimeout(() => {
                consentBox.classList.remove('highlight-attention');
            }, 2000); 
        }, 100); 
    }

    // --- Control Button Event Listeners ---
    searchBtn.addEventListener('click', () => {
        if (currentFile) {
            // Hide controls while searching
            uploadControls.classList.add('hidden'); 
            uploadAndSearch(currentFile);
        }
    });
    removeBtn.addEventListener('click', () => { resetApp(); });
    startAgainBtn.addEventListener('click', () => { resetApp(); });
    
    // Copy UUID Logic
    copyUuidBtn.addEventListener('click', () => {
        const uuid = userUuid.textContent;
        if (navigator.clipboard) {
            navigator.clipboard.writeText(uuid).then(() => {
                copyUuidBtn.textContent = 'Copied!';
                copyUuidBtn.classList.add('copied');
                setTimeout(() => {
                    copyUuidBtn.textContent = 'Copy';
                    copyUuidBtn.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                console.error('Failed to copy UUID: ', err);
                alert('Failed to copy. Please copy manually.');
            });
        }
    });

    // --- Carousel Event Listeners ---
    carouselNext.addEventListener('click', () => {
        const maxIndex = Math.max(0, totalItems - itemsPerPage); 
        if (currentIndex < maxIndex) {
            currentIndex++; 
            updateCarouselPosition();
        }
    });

    carouselPrev.addEventListener('click', () => {
        if (currentIndex > 0) {
            currentIndex--; 
            updateCarouselPosition();
        }
    });

    // --- About Page CTA Listener ---
    if (ctaTryNowBtn) {
        ctaTryNowBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const searchLink = document.querySelector('.nav-link[data-tab="search"]');
            if (searchLink) searchLink.click();
        });
    }

    // --- Privacy Link Listeners ---
    function handlePrivacyLinkClick(e) {
        e.preventDefault();
        const aboutLink = document.querySelector('.nav-link[data-tab="about"]');
        if (aboutLink) aboutLink.click();
        setTimeout(() => {
            const privacyBox = document.querySelector('.privacy-highlight-box');
            if (privacyBox) {
                privacyBox.scrollIntoView({ behavior: 'smooth', block: 'center' });
                privacyBox.classList.add('highlight-attention');
                setTimeout(() => privacyBox.classList.remove('highlight-attention'), 2000);
            }
        }, 100);
    }
    if (consentPrivacyLink) {
        consentPrivacyLink.addEventListener('click', handlePrivacyLinkClick);
    }
    
    function handleManageDataLinkClick(e) {
        e.preventDefault();
        const manageDataLink = document.querySelector('.nav-link[data-tab="manage-data"]');
        if (manageDataLink) manageDataLink.click();
    }
    if (aboutPrivacyLink) {
        aboutPrivacyLink.addEventListener('click', handleManageDataLinkClick);
    }

    // --- Image Preloader Function ---
    function preloadImages(urls) {
        const promises = urls.map(url => {
            return new Promise((resolve) => {
                const img = new Image();
                img.onload = () => resolve(img);
                img.onerror = () => {
                    console.error("Failed to preload image:", url);
                    resolve(null); 
                };
                img.src = url;
            });
        });
        return Promise.all(promises);
    }

    // --- API Search Function ---
    async function uploadAndSearch(file) {
        console.log('Uploading file:', file.name);

        resultsSection.classList.remove('hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
        
        loader.classList.remove('hidden'); 
        
        // Hide result containers initially
        topResultContainer.classList.add('hidden');
        carouselContainer.classList.add('hidden'); 
        otherResultsTitle.classList.add('hidden'); 
        searchErrorMessage.classList.add('hidden');
        carouselTrack.innerHTML = ''; 
        
        resetControls.classList.add('hidden');
        saveInfo.classList.add('hidden'); 

        const formData = new FormData();
        formData.append("file", file);
        formData.append("consent", consentCheckbox.checked); 

        try {
            const response = await fetch(`${API_BASE_URL}/search/`, {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
                const errData = await response.json();
                
                // Check specifically for the "No face detected" 400 error
                if (response.status === 400) {
                    throw new Error("NO_FACE"); 
                }
                
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            if (!data.results || data.results.length === 0) {
                console.log("No results returned from API.");
                displayResults([]); 
                
                loader.classList.add('hidden'); 
            
                resetControls.classList.remove('hidden'); 
                return;
            }

            // Fix URLs
            const fixedResults = data.results.map(r => {
                if (r.url.startsWith('/')) {
                    return { ...r, url: `${API_BASE_URL}${r.url}` };
                }
                return r;
            });

            const imageUrls = fixedResults.map(result => result.url);
            console.log("Preloading images...", imageUrls);
            await preloadImages(imageUrls);
            console.log("All images preloaded.");
            
            // --- FIX: Hide loader before showing results ---
            loader.classList.add('hidden');
            
            displayResults(fixedResults); 

            if (data.uuid) {
                saveInfo.classList.remove('hidden');
                userUuid.textContent = data.uuid;
            }

            resetControls.classList.remove('hidden');

        } catch (error) {
            console.error("Error during search:", error);
            
            // 1. Hide the loader
            loader.classList.add('hidden');
            
            // 2. Hide the content parts of the results section
            topResultContainer.classList.add('hidden');
            carouselContainer.classList.add('hidden');
            otherResultsTitle.classList.add('hidden');

            // 3. Keep the results section wrapper visible so "Start Again" can be seen
            resultsSection.classList.remove('hidden'); 

            // 4. Handle Specific Errors
            if (error.message === "NO_FACE") {
                searchErrorMessage.classList.remove('hidden');
                errorText.textContent = "We couldn't find a face in this image. Please try uploading a photo with a clear, visible face.";
                
                // Scroll to the error
                searchErrorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                // Generic error fallback
                searchErrorMessage.classList.remove('hidden');
                errorText.textContent = `Search failed: ${error.message}`;
                searchErrorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            uploadControls.classList.add('hidden');   
            resetControls.classList.remove('hidden'); 
        }
    }

    // --- Display Functions ---
    function displayResults(results) {
        carouselTrack.innerHTML = ''; 

        if (!results || results.length === 0) {
            otherResultsTitle.textContent = 'No matching results found.';
            otherResultsTitle.classList.remove('hidden');
            topResultContainer.classList.add('hidden');
            carouselContainer.classList.add('hidden');
            return;
        }
        
        otherResultsTitle.textContent = 'Other Similar Results';

        const topResult = results[0];
        topResultContainer.classList.remove('hidden');
        topResultImageWrapper.innerHTML = `<img src="${topResult.url}" alt="Top Similar Image" data-fullsrc="${topResult.url}">`;
        animateCount(matchPercentageNumber, topResult.similarity, 1000); 
        setMatchPercentageStyle(matchPercentageNumber, topResult.similarity);
        topResultImageWrapper.querySelector('img').addEventListener('click', () => {
            openModal(topResult.url);
        });

        const otherResults = results.slice(1);
        totalItems = otherResults.length;
        currentIndex = 0; 

        if (totalItems === 0) {
            otherResultsTitle.classList.add('hidden');
            carouselContainer.classList.add('hidden');
            return;
        }
        
        otherResultsTitle.classList.remove('hidden');
        carouselContainer.classList.remove('hidden');
        
        otherResults.forEach(result => {
            const resultItem = document.createElement('div');
            resultItem.classList.add('result-item');

            const img = document.createElement('img');
            img.src = result.url;
            img.alt = 'Similar Image Result';
            img.dataset.fullsrc = result.url; 

            const matchSpan = document.createElement('span');
            matchSpan.classList.add('result-match-small');
            matchSpan.textContent = `${result.similarity}%`;
            setMatchPercentageStyle(matchSpan, result.similarity); 

            resultItem.appendChild(img);
            resultItem.appendChild(matchSpan);
            
            img.addEventListener('click', () => {
                openModal(result.url);
            });
            
            carouselTrack.appendChild(resultItem);
        });
        
        carouselTrack.style.width = null; 

        updateCarouselPosition(); 
    }

    // --- Carousel Helper Functions ---
    function updateCarouselPosition() {
        const safeItemsPerPage = Math.max(1, itemsPerPage);
        const itemWidthPercent = 100 / safeItemsPerPage;
        
        const maxIndex = Math.max(0, totalItems - safeItemsPerPage);
        currentIndex = Math.max(0, Math.min(currentIndex, maxIndex));
        
        const newTransform = -currentIndex * itemWidthPercent;
        carouselTrack.style.transform = `translateX(${newTransform}%)`;
        updateCarouselControls();
    }
    
    function updateCarouselControls() {
        const maxIndex = Math.max(0, totalItems - itemsPerPage);
        if (totalItems <= itemsPerPage) {
            carouselPrev.classList.add('hidden');
            carouselNext.classList.add('hidden');
        } else {
            carouselPrev.classList.toggle('hidden', currentIndex === 0);
            carouselNext.classList.toggle('hidden', currentIndex >= maxIndex);
        }
    }
    
    function animateCount(element, target, duration) {
        let start = 0;
        const finalTarget = parseInt(target, 10) || 0;
        const startTime = performance.now();

        function step(currentTime) {
            const elapsedTime = currentTime - startTime;
            const progress = Math.min(elapsedTime / duration, 1);
            
            const easedProgress = progress * (2 - progress);
            
            const current = Math.floor(easedProgress * finalTarget);
            element.textContent = current;

            if (progress < 1) {
                requestAnimationFrame(step);
            } else {
                element.textContent = finalTarget;
            }
        }
        requestAnimationFrame(step);
    }
    
    function setMatchPercentageStyle(element, percentage) {
        element.classList.remove('match-green', 'match-orange', 'match-red');
        if (percentage > 80) element.classList.add('match-green');
        else if (percentage >= 50) element.classList.add('match-orange');
        else element.classList.add('match-red');
    }

    // --- Reset Function ---
    function resetApp() {
        currentFile = null;
        fileInput.value = null; 
        uploadPreview.innerHTML = '';
        dropZone.classList.remove('file-loaded');
        dropZone.classList.remove('dragover');
        uploadControls.classList.add('hidden');
        consentCheckbox.checked = false; 
        
        resultsSection.classList.add('hidden');
        topResultContainer.classList.add('hidden');
        carouselContainer.classList.add('hidden');
        otherResultsTitle.classList.add('hidden');
        otherResultsTitle.textContent = 'Other Similar Results'; 
        
        carouselTrack.innerHTML = '';
        carouselTrack.style.transform = 'translateX(0%)'; 
        currentIndex = 0; 
        totalItems = 0;
        
        resetControls.classList.add('hidden');
        loader.classList.add('hidden');
        saveInfo.classList.add('hidden'); 
        userUuid.textContent = ''; 
        
        copyUuidBtn.textContent = 'Copy';
        copyUuidBtn.classList.remove('copied');
        
        matchPercentageNumber.textContent = '0';
        matchPercentageNumber.classList.remove('match-green', 'match-orange', 'match-red');
        
        window.scrollTo({ top: 0, behavior: 'smooth' });

        searchErrorMessage.classList.add('hidden');
    }

    // --- Image Modal Functionality ---
    function openModal(imageSrc) {
        modalImage.src = imageSrc;
        imageModal.classList.add('active');
        imageModal.classList.remove('hidden');
        document.body.style.overflow = 'hidden';
    }
    function closeModal() {
        imageModal.classList.add('hidden');
        imageModal.classList.remove('active');
        modalImage.src = '';
        document.body.style.overflow = '';
    }
    modalCloseBtn.addEventListener('click', closeModal);
    imageModal.addEventListener('click', (e) => {
        if (e.target === imageModal) closeModal();
    });
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && imageModal.classList.contains('active')) closeModal();
    });

    // --- DELETE FORM LOGIC ---
    if (deleteForm) {
        deleteForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const uuid = uuidInput.value.trim();
            if (!uuid) {
                showDeleteMessage("Please enter a valid UUID.", "error");
                return;
            }

            deleteBtn.classList.add('loading');
            deleteBtn.querySelector('.btn-text').classList.add('hidden');
            deleteBtn.querySelector('.btn-loader').classList.remove('hidden');
            deleteBtn.disabled = true;
            deleteMessage.classList.add('hidden');

            try {
                const response = await fetch(`${API_BASE_URL}/delete/${uuid}`, {
                    method: 'DELETE',
                });

                const data = await response.json();

                if (response.ok) {
                    showDeleteMessage(data.message, 'success');
                    uuidInput.value = ''; 
                } else {
                    throw new Error(data.detail || "An unknown error occurred.");
                }

            } catch (error) {
                console.error("Error deleting file:", error);
                showDeleteMessage(error.message, 'error');
            } finally {
                deleteBtn.classList.remove('loading');
                deleteBtn.querySelector('.btn-text').classList.remove('hidden');
                deleteBtn.querySelector('.btn-loader').classList.add('hidden');
                deleteBtn.disabled = false;
            }
        });
    }

    function showDeleteMessage(message, type) {
        deleteMessage.textContent = message;
        deleteMessage.classList.remove('hidden', 'success', 'error');
        deleteMessage.classList.add(type); 
    }

    // --- Scroll Animations for About Page ---
    const animatedElements = document.querySelectorAll('.animate-on-scroll');
    if ("IntersectionObserver" in window) {
        const observer = new IntersectionObserver((entries, obs) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.classList.add('is-visible');
                    obs.unobserve(entry.target);
                }
            });
        }, { threshold: 0.1 });
        animatedElements.forEach(el => observer.observe(el));
    } else {
        animatedElements.forEach(el => el.classList.add('is-visible'));
    }
});