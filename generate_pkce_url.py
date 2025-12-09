# generate_pkce_url.py
import base64, hashlib, secrets, urllib.parse

def b64url(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()

# 1. Generate verifier & challenge
code_verifier = b64url(secrets.token_bytes(64))
code_challenge = b64url(hashlib.sha256(code_verifier.encode()).digest())

# 2. Save verifier (you’ll use it later)
print("Your code_verifier (SAVE THIS):")
print(code_verifier)
print()

# 3. Build authorize URL
CLIENT_ID = "9EiCwuDx9UxzffPESDS8royzcH815LfZ"
REDIRECT_URI = "http://localhost:8501/"
SCOPES = "offline_access read:account read:me read:content:confluence write:content:confluence read:space-details:confluence write:attachment:confluence"

params = {
    "audience": "api.atlassian.com",
    "client_id": CLIENT_ID,
    "scope": SCOPES,
    "redirect_uri": REDIRECT_URI,
    "response_type": "code",
    "prompt": "consent",
    "code_challenge": code_challenge,
    "code_challenge_method": "S256",
}

authorize_url = "https://auth.atlassian.com/authorize?" + urllib.parse.urlencode(params)
print("Open this URL in your browser:")
print(authorize_url)
