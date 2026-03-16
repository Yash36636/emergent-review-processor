"""
Vercel Python entrypoint — imports Flask app for serverless deployment.
"""
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

handler = app
