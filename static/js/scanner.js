/**
 * scanner.js — PhishNet v2.0
 * Handles UI interactions, loading animations, and copy-to-clipboard.
 */

document.addEventListener('DOMContentLoaded', () => {
    const scanForm = document.getElementById('scanForm');
    const loadingOverlay = document.getElementById('loadingOverlay');
    const progressBar = document.getElementById('progressBar');
    const progressText = document.getElementById('progressText');
    const scanBtn = document.getElementById('scanBtn');

    if (scanForm) {
        scanForm.addEventListener('submit', (e) => {
            // Show loading overlay
            loadingOverlay.style.display = 'flex';
            scanBtn.disabled = true;
            scanBtn.innerText = 'Scanning...';

            let progress = 0;
            const interval = setInterval(() => {
                progress += Math.random() * 15;
                if (progress >= 100) {
                    progress = 100;
                    clearInterval(interval);
                    // The form will naturally submit and reload the page
                }
                progressBar.style.width = `${progress}%`;
                progressText.innerText = `Analysing URL security layers... ${Math.round(progress)}%`;
            }, 100);
        });
    }
});

/**
 * Copy URL to clipboard helper
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        const btn = document.getElementById('copyBtn');
        const originalText = btn.innerHTML;
        btn.innerHTML = '✅ Copied!';
        btn.classList.replace('btn-navy', 'btn-green');
        setTimeout(() => {
            btn.innerHTML = originalText;
            btn.classList.replace('btn-green', 'btn-navy');
        }, 2000);
    });
}
