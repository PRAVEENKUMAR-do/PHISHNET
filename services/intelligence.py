# services/intelligence.py — PhishNet Real-World Signal Checks
# Provides WHOIS, SSL, DNS signals to SUPPORT (not replace) the ML model.
import re
import ssl
import socket
import datetime
from urllib.parse import urlparse

# ── Optional imports with graceful fallbacks ──────────────────
try:
    import whois as _whois
    _WHOIS_AVAILABLE = True
except ImportError:
    _WHOIS_AVAILABLE = False

try:
    import dns.resolver as _dns_resolver
    _DNS_AVAILABLE = True
except ImportError:
    _DNS_AVAILABLE = False


# ─────────────────────────────────────────────────────────────
# TASK 1 — Domain Extraction
# ─────────────────────────────────────────────────────────────

def get_domain(url: str) -> str:
    """Extract clean hostname from any URL format."""
    url = url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    try:
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        # Strip port if present
        host = host.split(":")[0]
        return host
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────
# TASK 2 — Domain Age via WHOIS
# ─────────────────────────────────────────────────────────────

def get_domain_age(domain: str) -> int:
    """
    Returns age of domain in days using WHOIS lookup.
    Returns -1 if lookup fails (WHOIS privacy, network timeout, etc.)
    """
    if not _WHOIS_AVAILABLE or not domain:
        return -1
    try:
        info = _whois.whois(domain)
        creation = info.creation_date

        # creation_date can be a list or a single datetime
        if isinstance(creation, list):
            creation = creation[0]

        if creation is None:
            return -1

        # Ensure it's a datetime object
        if isinstance(creation, str):
            # Attempt to parse ISO format
            creation = datetime.datetime.fromisoformat(creation)

        now = datetime.datetime.now()
        age_days = (now - creation).days
        return max(age_days, 0)

    except Exception:
        return -1


# ─────────────────────────────────────────────────────────────
# TASK 3 — SSL Certificate Check
# ─────────────────────────────────────────────────────────────

def check_ssl(domain: str, timeout: int = 5) -> bool:
    """
    Attempts TLS handshake with the domain on port 443.
    Returns True if a valid certificate is presented, False otherwise.
    Errors are caught silently — a failing check returns False.
    """
    if not domain:
        return False
    try:
        ctx = ssl.create_default_context()
        with ctx.wrap_socket(
            socket.create_connection((domain, 443), timeout=timeout),
            server_hostname=domain
        ):
            return True
    except Exception:
        return False


# ─────────────────────────────────────────────────────────────
# TASK 4 — DNS Resolution Check
# ─────────────────────────────────────────────────────────────

def check_dns(domain: str) -> bool:
    """
    Resolves A record for domain using dnspython if available,
    falls back to socket.getaddrinfo for environments without dnspython.
    Returns True if any record resolves, False otherwise.
    """
    if not domain:
        return False

    if _DNS_AVAILABLE:
        try:
            _dns_resolver.resolve(domain, "A")
            return True
        except Exception:
            return False
    else:
        # Fallback: standard socket lookup
        try:
            socket.getaddrinfo(domain, None)
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────────────────────
# TASK 5 — Smart Heuristic Risk Scoring
# ─────────────────────────────────────────────────────────────

def compute_intelligence_signals(url: str) -> dict:
    """
    Runs all real-world intelligence checks for a URL.
    Returns a structured dict with signals AND a supporting risk score + factors.
    This SUPPORTS the ML model — it does NOT override ML prediction.
    """
    domain    = get_domain(url)
    age_days  = get_domain_age(domain)
    ssl_valid = check_ssl(domain)
    dns_valid = check_dns(domain)

    # ── Smart Rule-Based Risk Scoring ────────────────────────
    risk_score   = 0
    risk_factors = []

    # Domain age signal
    if age_days != -1 and age_days < 30:
        risk_score += 2
        risk_factors.append("Domain registered recently (< 30 days)")
    
    # SSL signal — only *absence* of SSL is a risk flag
    # HTTPS presence itself is NOT treated as safety (per requirements)
    if not ssl_valid:
        risk_score += 2
        risk_factors.append("No valid SSL certificate")

    # DNS signal
    if not dns_valid:
        risk_score += 1
        risk_factors.append("DNS resolution failed")

    # URL structure signals
    if "@" in url:
        risk_score += 2
        risk_factors.append("Contains @-symbol redirect")

    if len(url) > 75:
        risk_score += 1
        risk_factors.append("Unusually long URL")

    return {
        "domain":       domain,
        "age_days":     age_days,
        "age_display":  f"{age_days} days" if age_days >= 0 else "Unavailable",
        "ssl_valid":    ssl_valid,
        "dns_valid":    dns_valid,
        "intel_score":  risk_score,          # supporting signal only
        "intel_factors": risk_factors,        # supporting factors only
    }
