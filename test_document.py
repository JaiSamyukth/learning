#!/usr/bin/env python3
"""
Create a test PDF document for testing the upload functionality.
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import os

def create_test_pdf():
    """Create a simple test PDF document"""
    filename = "test_document.pdf"
    
    # Create a canvas
    c = canvas.Canvas(filename, pagesize=letter)
    width, height = letter
    
    # Add title
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, height - 100, "Test Document for PDF Upload")
    
    # Add some content
    c.setFont("Helvetica", 12)
    y_position = height - 150
    
    content = [
        "This is a test PDF document created for testing the PDF upload functionality.",
        "",
        "Chapter 1: Introduction",
        "This document contains sample text to test the PDF text extraction capabilities.",
        "The system should be able to extract this text and process it for AI-powered",
        "question generation and chat functionality.",
        "",
        "Chapter 2: Features",
        "- PDF text extraction",
        "- AI-powered question generation", 
        "- Interactive chat with document content",
        "- Intelligent learning assistance",
        "",
        "Chapter 3: Testing",
        "This document serves as a test case to verify that:",
        "1. PDF files can be uploaded successfully",
        "2. Text extraction works correctly",
        "3. The logging system functions properly",
        "4. No errors occur during the upload process",
        "",
        "Conclusion:",
        "If you can read this text after uploading, the system is working correctly!"
    ]
    
    for line in content:
        if line:  # Skip empty lines for spacing
            c.drawString(100, y_position, line)
        y_position -= 20
        
        # Start new page if needed
        if y_position < 100:
            c.showPage()
            c.setFont("Helvetica", 12)
            y_position = height - 100
    
    # Save the PDF
    c.save()
    print(f"Test PDF created: {filename}")
    return filename

if __name__ == "__main__":
    create_test_pdf()
