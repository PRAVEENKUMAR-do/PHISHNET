import sys
import os

# Add the project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.detector import analyse

def test_whitelist():
    test_urls = [
        "https://www.google.com",
        "github.com/microsoft",
        "https://stackoverflow.com/questions",
        "amazon.in/p/123",
        "netflix.com",
        "not-a-whitelisted-domain.xyz"
    ]
    
    print(f"{'URL':<40} | {'Risk':<5} | {'Malicious':<10} | {'Verdict'}")
    print("-" * 75)
    
    for url in test_urls:
        res = analyse(url)
        print(f"{url:<40} | {res['risk_score']:<5} | {str(res['is_malicious']):<10} | {res['prediction']}")

if __name__ == "__main__":
    test_whitelist()
