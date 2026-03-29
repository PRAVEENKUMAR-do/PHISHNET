# src/features.py  —  PhishNet
# Uses smarter features that correctly separate benign from malicious
import re, math
from urllib.parse import urlparse

FEATURE_NAMES = [
    "url_length",
    "num_dots",
    "num_hyphens",
    "num_at",
    "num_question",
    "num_equal",
    "num_underscore",
    "num_percent",
    "num_slash",
    "num_digits",
    "digit_ratio",
    "letter_ratio",
    "url_entropy",
    "has_ip",
    "has_https",
    "has_suspicious_keyword",
    "subdomain_depth",
    "path_depth",
    "domain_length",
    "is_shortened",
    "has_at_symbol",
    "has_double_slash_in_path",
    "has_hex_chars",
    "special_char_ratio",
    "tld_in_middle",
]

SHORTENERS = {
    "bit.ly","goo.gl","tinyurl.com","t.co","ow.ly","is.gd",
    "buff.ly","adf.ly","tiny.cc","bc.vc","su.pr","twurl.nl",
    "snipurl.com","short.to","budurl.com","ping.fm","post.ly",
    "just.as","bkite.com","snipr.com","fic.kr","loopt.us"
}

BAD_KEYWORDS = [
    "login","signin","verify","update","secure","account",
    "banking","confirm","password","credential","free","lucky",
    "winner","click","alert","invoice","suspended","unusual",
    "access","recover","restore","validate","submit","redirect",
    "buy","offer","limited","expire","unlock","urgent",
]

SUSPICIOUS_TLDS = {
    "tk","ml","ga","cf","gq","pw","xyz","top","club","online",
    "site","work","click","link","rocks","party","racing",
    "win","loan","faith","review","trade","date","cricket",
    "bid","webcam","accountant","science","download","stream"
}

def _entropy(s):
    if not s:
        return 0.0
    p = [float(s.count(c)) / len(s) for c in set(s)]
    return -sum(x * math.log2(x) for x in p if x > 0)

def extract_features(raw_url: str) -> list:
    url = str(raw_url).strip()
    if not url:
        return [0.0] * len(FEATURE_NAMES)
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    try:
        parsed = urlparse(url)
    except Exception:
        return [0.0] * len(FEATURE_NAMES)

    full   = url
    scheme = parsed.scheme or ""
    netloc = parsed.netloc or ""
    path   = parsed.path or ""
    query  = parsed.query or ""
    host   = netloc.lower().split(":")[0]

    # domain parts
    host_parts = host.split(".")
    tld        = host_parts[-1] if len(host_parts) >= 1 else ""
    domain     = host_parts[-2] if len(host_parts) >= 2 else host
    subdomains  = host_parts[:-2] if len(host_parts) > 2 else []
    sub_depth  = len(subdomains)

    L = max(len(full), 1)

    # ── basic counts ──────────────────────────────────────────
    n_dots   = full.count(".")
    n_hyph   = full.count("-")
    n_at     = full.count("@")
    n_q      = full.count("?")
    n_eq     = full.count("=")
    n_und    = full.count("_")
    n_pct    = full.count("%")
    n_slash  = full.count("/")
    n_digits = sum(1 for c in full if c.isdigit())
    n_alpha  = sum(1 for c in full if c.isalpha())
    n_spec   = sum(1 for c in full if c in "!#$^&*()+=[]{}|;<>,~`")

    # ── IP in host ────────────────────────────────────────────
    ip_pat = r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
    has_ip = 1.0 if re.match(ip_pat, host) else 0.0

    # ── HTTPS ─────────────────────────────────────────────────
    has_https = 1.0 if scheme == "https" else 0.0

    # ── suspicious keyword in full URL ────────────────────────
    url_lower = full.lower()
    has_kw = 1.0 if any(k in url_lower for k in BAD_KEYWORDS) else 0.0

    # ── URL entropy ───────────────────────────────────────────
    entropy = _entropy(full)

    # ── path depth ───────────────────────────────────────────
    path_parts = [p for p in path.split("/") if p]
    path_depth = len(path_parts)

    # ── shortened URL ────────────────────────────────────────
    is_short = 1.0 if host in SHORTENERS else 0.0

    # ── double slash in path (redirect trick) ────────────────
    has_ds = 1.0 if "//" in path else 0.0

    # ── hex encoded chars (%XX) ──────────────────────────────
    has_hex = 1.0 if re.search(r"%[0-9a-fA-F]{2}", full) else 0.0

    # ── suspicious TLD in the MIDDLE of domain ───────────────
    # e.g. paypal.com.evil.tk  — "com" appears before the real TLD
    middle_parts = host_parts[:-1]  # everything except actual TLD
    tld_mid = 1.0 if any(p in {"com","net","org","gov","edu"}
                          for p in middle_parts[:-1]) else 0.0

    return [
        float(len(full)),            # url_length
        float(n_dots),               # num_dots
        float(n_hyph),               # num_hyphens
        float(n_at),                 # num_at
        float(n_q),                  # num_question
        float(n_eq),                 # num_equal
        float(n_und),                # num_underscore
        float(n_pct),                # num_percent
        float(n_slash),              # num_slash
        float(n_digits),             # num_digits
        float(n_digits) / L,         # digit_ratio
        float(n_alpha) / L,          # letter_ratio
        float(entropy),              # url_entropy
        has_ip,                      # has_ip
        has_https,                   # has_https
        has_kw,                      # has_suspicious_keyword
        float(sub_depth),            # subdomain_depth
        float(path_depth),           # path_depth
        float(len(domain)),          # domain_length
        is_short,                    # is_shortened
        float(n_at),                 # has_at_symbol
        has_ds,                      # has_double_slash_in_path
        has_hex,                     # has_hex_chars
        float(n_spec) / L,           # special_char_ratio
        tld_mid,                     # tld_in_middle
    ]
