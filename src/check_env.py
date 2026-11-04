import os
from dotenv import load_dotenv

# reads env file into env var
load_dotenv() 
token = os.getenv("GITHUB_TOKEN")

if token:
    print(f"Token loaded. Starts with: {token[:11]}")
else:
    print("Token not found")