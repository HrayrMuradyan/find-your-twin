document.addEventListener('DOMContentLoaded', () => {
    
    // Production URLs
    const PROD_INFERENCE_URL = "https://hrayrmuradyan-find-your-twin-inference.hf.space";
    const PROD_DATABASE_URL = "https://hrayrmuradyan-find-your-twin-database.hf.space"; 

    // Local URLs
    const LOCAL_INFERENCE_URL = "http://127.0.0.1:7860";
    const LOCAL_DATABASE_URL = "http://127.0.0.1:8800";

    // Environment Detection
    const isLocalEnvironment = 
        window.location.hostname === 'localhost' || 
        window.location.hostname === '127.0.0.1' || 
        window.location.protocol === 'file:';

    const API_BASE_URL = isLocalEnvironment ? LOCAL_INFERENCE_URL : PROD_INFERENCE_URL;
    const DB_BASE_URL = isLocalEnvironment ? LOCAL_DATABASE_URL : PROD_DATABASE_URL;

    console.log(`Environment detected: ${isLocalEnvironment ? 'DEVELOPMENT' : 'PRODUCTION'}`);

    const mobileMenuBtn = document.getElementById('mobile-menu-btn');
    const navLinksContainer = document.querySelector('.nav-links'); // Using class to target nav
    let menuIcon = null;

    if (mobileMenuBtn) {
        menuIcon = mobileMenuBtn.querySelector('i');
    }

    function toggleMenu() {
        if (!navLinksContainer) return;
        
        navLinksContainer.classList.toggle('active');
        
        if (menuIcon) {
            // Toggle icon between 'menu' and 'x'
            if (navLinksContainer.classList.contains('active')) {
                menuIcon.classList.remove('bx-menu');
                menuIcon.classList.add('bx-x');
            } else {
                menuIcon.classList.remove('bx-x');
                menuIcon.classList.add('bx-menu');
            }
        }
    }

    // Toggle on button click
    if (mobileMenuBtn) {
        mobileMenuBtn.addEventListener('click', (e) => {
            e.stopPropagation(); 
            toggleMenu();
        });
    }

    // Close menu when a link is clicked
    const allNavLinks = document.querySelectorAll('.nav-link');
    allNavLinks.forEach(link => {
        link.addEventListener('click', () => {
            if (navLinksContainer && navLinksContainer.classList.contains('active')) {
                toggleMenu();
            }
        });
    });

    // Close menu if clicking outside of it
    document.addEventListener('click', (e) => {
        if (navLinksContainer && navLinksContainer.classList.contains('active') && 
            !navLinksContainer.contains(e.target) && 
            mobileMenuBtn && !mobileMenuBtn.contains(e.target)) {
            toggleMenu();
        }
    });

    // Startup & Health Check Logic
    
    const startupOverlay = document.getElementById('startup-overlay');
    const mainAppWrapper = document.getElementById('main-app-wrapper');
    const statusInference = document.getElementById('status-inference');
    const statusDb = document.getElementById('status-db');
    const dbStatsContainer = document.getElementById('db-stats-container');
    const dbCountNumber = document.getElementById('db-count-number');
    let hasInitialAnimationRun = false; 

    // Mark UI as active
    function markServiceActive(element) {
        if (!element) return;
        element.classList.add('active'); 
    }

    // Poll service until 200 OK
    async function checkServiceHealth(url, statusElement) {
        const pollInterval = 2000; 
        
        const check = async () => {
            try {
                // Using fetch with a timeout signal
                const response = await fetch(url, { 
                    method: 'GET',
                    signal: AbortSignal.timeout(5000) 
                });
                
                if (response.ok) {
                    markServiceActive(statusElement);
                    return true;
                }
            } catch (e) {
                console.log(`Waiting for ${url}...`);
            }
            // Retry
            await new Promise(r => setTimeout(r, pollInterval));
            return check();
        };
        return check();
    }

    // Main System Check
    async function initSystemCheck() {
        console.log("Starting System Health Checks...");
        
        document.body.style.overflow = 'hidden';

        // Inference health check
        const inferencePromise = checkServiceHealth(API_BASE_URL + "/", statusInference);
        
        // Database health check
        const dbPromise = checkServiceHealth(DB_BASE_URL + "/health", statusDb);

        await Promise.all([inferencePromise, dbPromise]);

        // Wait 2 seconds to show the green ticks
        setTimeout(() => {
            // Trigger CSS Animations
            startupOverlay.classList.add('fade-out');
            mainAppWrapper.classList.remove('blurred-content');
            mainAppWrapper.classList.add('content-visible');
            
            // Unlock scrolling after animation
            setTimeout(() => {
                document.body.style.overflow = ''; 
                fetchDatabaseStats(); 
            }, 500);
            
        }, 2000); 
    }

    // Start Check Immediately
    initSystemCheck();

    // Tab Switching
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
    
    // Uploader
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

    // Results
    const resultsSection = document.getElementById('results-section');
    const resultsHeader = resultsSection.querySelector('h3');
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

    // Carousel
    const carouselContainer = document.getElementById('carousel-container');
    const carouselTrack = document.getElementById('carousel-track');
    const carouselPrev = document.getElementById('carousel-prev');
    const carouselNext = document.getElementById('carousel-next');

    // Carousel State
    let currentIndex = 0; 
    let itemsPerPage = 5;
    let totalItems = 0;

    // Image Modal
    const imageModal = document.getElementById('image-modal');
    const modalImage = document.getElementById('modal-image');
    const modalCloseBtn = document.querySelector('.modal-close-btn');

    // About Page CTA
    const ctaTryNowBtn = document.getElementById('cta-try-now');
    const consentPrivacyLink = document.getElementById('consent-privacy-link'); 
    const aboutPrivacyLink = document.querySelector('.about-privacy-link'); 

    // Manage Data
    const deleteForm = document.getElementById('delete-form');
    const uuidInput = document.getElementById('uuid-input');
    const deleteBtn = document.getElementById('delete-btn');
    const deleteMessage = document.getElementById('delete-message');

    // Global State
    let currentFile = null;
    
    // Responsive Carousel Logic
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

    // Fetch DB Stats (Run after health check)
    async function fetchDatabaseStats() {
        if (!dbStatsContainer || !dbCountNumber) return;
        
        try {
            // FIX: Added timestamp to prevent browser caching
            const response = await fetch(`${API_BASE_URL}/stats?t=${new Date().getTime()}`);
            
            if (!response.ok) return;

            const data = await response.json();
            const finalCount = data.count || 0;

            dbStatsContainer.classList.add('stats-visible');

            if (!hasInitialAnimationRun) {
                // First Startup
                animateValueWithCommas(dbCountNumber, 0, finalCount, 2000, () => {
                    dbCountNumber.classList.add('count-pop');
                });
                hasInitialAnimationRun = true;
            } else {
                // Update after upload
                const currentVal = parseInt(dbCountNumber.textContent.replace(/,/g, '')) || 0;
                
                // Only update if count has changed
                if (finalCount !== currentVal) {
                    dbCountNumber.innerHTML = finalCount.toLocaleString();
                    dbCountNumber.classList.remove('count-pop');
                    void dbCountNumber.offsetWidth; 
                    dbCountNumber.classList.add('count-pop');
                }
            }
            
        } catch (error) {
            console.warn("Could not fetch DB stats:", error);
        }
    }
    
    function animateValueWithCommas(obj, start, end, duration, onComplete) {
        let startTimestamp = null;
        const step = (timestamp) => {
            if (!startTimestamp) startTimestamp = timestamp;
            const progress = Math.min((timestamp - startTimestamp) / duration, 1);
            const easeProgress = 1 - (1 - progress) * (1 - progress);
            const currentVal = Math.floor(easeProgress * (end - start) + start);
            
            obj.innerHTML = currentVal.toLocaleString();
            
            if (progress < 1) {
                window.requestAnimationFrame(step);
            } else {
                obj.innerHTML = end.toLocaleString();
                if (onComplete) onComplete();
            }
        };
        window.requestAnimationFrame(step);
    }

    // Uploader Event Listeners 
    browseBtn.addEventListener('click', (e) => {
        e.stopPropagation(); 
        fileInput.click();
    });

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

    // File Handling 
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

    // Control Button Event Listeners 
    searchBtn.addEventListener('click', () => {
        if (currentFile) {
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

    // Carousel Event Listeners 
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

    // About Page CTA Listener 
    if (ctaTryNowBtn) {
        ctaTryNowBtn.addEventListener('click', (e) => {
            e.preventDefault();
            const searchLink = document.querySelector('.nav-link[data-tab="search"]');
            if (searchLink) searchLink.click();
        });
    }

    // Privacy Link Listeners 
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

    // Image Preloader Function 
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

    // API Search Function 
    async function uploadAndSearch(file) {
        console.log('Uploading file:', file.name);

        resultsSection.classList.remove('hidden');
        resultsHeader.classList.remove('hidden'); 
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
                if (response.status === 400) {
                    throw new Error("NO_FACE"); 
                }
                throw new Error(errData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();

            // Refresh stats if we consented
            if (consentCheckbox.checked && data.uuid) {
                setTimeout(fetchDatabaseStats, 1000);
            }

            if (!data.results || data.results.length === 0) {
                console.log("No results returned from API.");
                displayResults([]); 
                loader.classList.add('hidden'); 
                resetControls.classList.remove('hidden'); 
                return;
            }

            // Fix URLs if relative
            const fixedResults = data.results.map(r => {
                if (r.url.startsWith('/')) {
                    return { ...r, url: `${API_BASE_URL}${r.url}` };
                }
                return r;
            });

            const imageUrls = fixedResults.map(result => result.url);
            console.log("Preloading images...", imageUrls);
            await preloadImages(imageUrls);
            
            loader.classList.add('hidden');
            displayResults(fixedResults); 

            if (data.uuid) {
                saveInfo.classList.remove('hidden');
                userUuid.textContent = data.uuid;
            }

            resetControls.classList.remove('hidden');

        } catch (error) {
            console.error("Error during search:", error);
            
            loader.classList.add('hidden');
            topResultContainer.classList.add('hidden');
            carouselContainer.classList.add('hidden');
            otherResultsTitle.classList.add('hidden');

            resultsSection.classList.remove('hidden'); 

            if (error.message === "NO_FACE") {
                resultsHeader.classList.add('hidden');
                searchErrorMessage.classList.remove('hidden');
                errorText.textContent = "We couldn't find a face in this image. Please try uploading a photo with a clear, visible face.";
                searchErrorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
            } else {
                searchErrorMessage.classList.remove('hidden');
                errorText.textContent = `Search failed: ${error.message}`;
                searchErrorMessage.scrollIntoView({ behavior: 'smooth', block: 'center' });
            }
            
            uploadControls.classList.add('hidden');   
            resetControls.classList.remove('hidden'); 
        }
    }

    // Display Functions 
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

    // Carousel Helper Functions 
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

    // Reset Function 
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

    // Image Modal Functionality 
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

    // Delete form logic
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
                    setTimeout(fetchDatabaseStats, 1000);
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

    // Scroll Animations 
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