#!/usr/bin/env python3
"""
Generate QR code for the Temporal GFN repository.
"""

import qrcode
from PIL import Image

# Repository URL
REPO_URL = "https://github.com/vincehass/Temporal-Generative-Flow-Networks-Temporal-GFNs-"

def generate_qr_code():
    """Generate a QR code for the repository."""
    
    # Create QR code instance
    qr = qrcode.QRCode(
        version=1,  # Controls the size (1-40)
        error_correction=qrcode.constants.ERROR_CORRECT_H,  # High error correction
        box_size=10,  # Size of each box in pixels
        border=4,  # Border size in boxes
    )
    
    # Add data
    qr.add_data(REPO_URL)
    qr.make(fit=True)
    
    # Create image
    img = qr.make_image(fill_color="black", back_color="white")
    
    # Save the QR code
    output_file = "repo_qr_code.png"
    img.save(output_file)
    
    print("=" * 60)
    print("✅ QR Code Generated Successfully!")
    print("=" * 60)
    print(f"\n📱 File: {output_file}")
    print(f"🔗 URL: {REPO_URL}")
    print("\n📋 Usage:")
    print("  • Scan with phone camera to open repository")
    print("  • Add to presentations/papers")
    print("  • Print on posters")
    print("  • Share on social media")
    print("\n" + "=" * 60)

def generate_qr_code_with_logo():
    """Generate a QR code with a custom center logo (optional)."""
    
    # Create QR code
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    
    qr.add_data(REPO_URL)
    qr.make(fit=True)
    
    # Create base QR image
    img = qr.make_image(fill_color="black", back_color="white").convert('RGB')
    
    # Save enhanced version
    output_file = "repo_qr_code_large.png"
    img = img.resize((800, 800), Image.Resampling.LANCZOS)
    img.save(output_file, quality=95)
    
    print(f"\n✅ Large QR Code: {output_file} (800x800)")

if __name__ == "__main__":
    print("\n🎨 Generating QR Codes for Temporal GFN Repository...\n")
    
    # Generate standard QR code
    generate_qr_code()
    
    # Generate large version
    generate_qr_code_with_logo()
    
    print("\n✨ Done! Scan the QR code to visit the repository.\n")

