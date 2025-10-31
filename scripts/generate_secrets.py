#!/usr/bin/env python3
# scripts/generate_secrets.py
"""
Generate secure secret keys for Flask and JWT.
Run this to generate new secrets for production deployment.

Usage:
    python scripts/generate_secrets.py
"""
import secrets
import sys
from pathlib import Path


def generate_secret(length=32):
    """Generate a secure random secret key"""
    return secrets.token_hex(length)


def main():
    """Generate and display secrets"""
    print("=" * 60)
    print("🔐 StuTally - Secret Key Generator")
    print("=" * 60)
    print()
    
    # Generate secrets
    flask_secret = generate_secret(32)
    jwt_secret = generate_secret(32)
    
    print("Generated secure secret keys:")
    print()
    print(f"SECRET_KEY={flask_secret}")
    print(f"JWT_SECRET_KEY={jwt_secret}")
    print()
    
    # Check if .env exists
    env_file = Path(__file__).parent.parent / '.env'
    env_example = Path(__file__).parent.parent / '.env.example'
    
    if not env_file.exists():
        print("ℹ️  .env file not found.")
        
        if env_example.exists():
            print(f"   Copy .env.example to .env and add the secrets above:")
            print(f"   cp {env_example} {env_file}")
        else:
            print(f"   Create .env file and add the secrets above:")
            print(f"   touch {env_file}")
        
        print()
    else:
        print(f"✅ .env file exists at: {env_file}")
        print()
        print("⚠️  WARNING: Update your .env file with these new secrets.")
        print("   Make sure to backup your old secrets before replacing them.")
        print()
    
    print("=" * 60)
    print("🔒 IMPORTANT: Keep these secrets safe!")
    print("   - Never commit them to git")
    print("   - Store them securely (password manager, vault)")
    print("   - Use different secrets for dev/staging/production")
    print("=" * 60)


if __name__ == '__main__':
    main()

