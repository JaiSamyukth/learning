"""
Create a comprehensive test book for RAG system testing
"""

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch

def create_comprehensive_test_book():
    """Create a comprehensive test book with rich content for RAG testing"""
    
    filename = "RAG_Test_Book_Advanced_AI_Concepts.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title = Paragraph("Advanced AI Concepts: A Comprehensive Guide to RAG Systems", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 0.5*inch))
    
    # Chapter 1: Introduction to RAG
    chapter1_title = Paragraph("Chapter 1: Introduction to Retrieval-Augmented Generation (RAG)", styles['Heading1'])
    story.append(chapter1_title)
    story.append(Spacer(1, 0.2*inch))
    
    chapter1_content = """
    Retrieval-Augmented Generation (RAG) is a revolutionary approach in artificial intelligence that combines 
    the power of large language models with external knowledge retrieval systems. This hybrid architecture 
    addresses one of the fundamental limitations of traditional language models: their reliance on static 
    training data and inability to access real-time or domain-specific information.
    
    The RAG system operates through a two-stage process: first, it retrieves relevant information from a 
    knowledge base using semantic search techniques, and then it generates responses by conditioning the 
    language model on both the user query and the retrieved context. This approach significantly improves 
    the accuracy, relevance, and factual correctness of AI-generated responses.
    
    Key components of a RAG system include:
    • Vector databases for efficient similarity search
    • Embedding models for semantic representation
    • Document chunking and preprocessing pipelines
    • Retrieval mechanisms with ranking algorithms
    • Generation models for response synthesis
    """
    
    para1 = Paragraph(chapter1_content, styles['Normal'])
    story.append(para1)
    story.append(Spacer(1, 0.3*inch))
    
    # Chapter 2: Vector Databases
    chapter2_title = Paragraph("Chapter 2: Vector Databases and Semantic Search", styles['Heading1'])
    story.append(chapter2_title)
    story.append(Spacer(1, 0.2*inch))
    
    chapter2_content = """
    Vector databases are specialized storage systems designed to handle high-dimensional vector embeddings 
    efficiently. In the context of RAG systems, they serve as the backbone for semantic search capabilities, 
    enabling the retrieval of contextually relevant information based on meaning rather than exact keyword matches.
    
    Popular vector database solutions include:
    • Qdrant: A high-performance vector search engine with advanced filtering capabilities
    • Pinecone: A managed vector database service optimized for machine learning applications
    • Weaviate: An open-source vector database with built-in vectorization modules
    • Chroma: A lightweight, embeddable vector database for AI applications
    
    The process of semantic search involves converting text into dense vector representations using embedding 
    models like sentence-transformers. These embeddings capture semantic meaning, allowing for similarity 
    calculations using metrics such as cosine similarity, Euclidean distance, or dot product.
    
    Qdrant, specifically, offers several advantages:
    • Rust-based implementation for high performance
    • Advanced filtering and payload support
    • Horizontal scaling capabilities
    • Rich API for integration with various programming languages
    • Cloud-hosted solutions for production deployments
    """
    
    para2 = Paragraph(chapter2_content, styles['Normal'])
    story.append(para2)
    story.append(Spacer(1, 0.3*inch))
    
    # Chapter 3: Document Processing
    chapter3_title = Paragraph("Chapter 3: Document Processing and Chunking Strategies", styles['Heading1'])
    story.append(chapter3_title)
    story.append(Spacer(1, 0.2*inch))
    
    chapter3_content = """
    Effective document processing is crucial for RAG system performance. The process involves several key steps:
    
    1. Text Extraction: Converting various document formats (PDF, DOCX, HTML) into plain text while preserving 
       structure and metadata.
    
    2. Chunking Strategy: Dividing large documents into smaller, manageable pieces that can be effectively 
       embedded and retrieved. Common approaches include:
       • Fixed-size chunking (e.g., 1000 characters with 200-character overlap)
       • Semantic chunking based on sentence or paragraph boundaries
       • Hierarchical chunking that preserves document structure
    
    3. Preprocessing: Cleaning text, removing noise, normalizing formatting, and handling special characters.
    
    4. Metadata Enrichment: Adding contextual information such as document source, creation date, section headers, 
       and topic classifications.
    
    The choice of chunking strategy significantly impacts retrieval quality. Smaller chunks provide more precise 
    matches but may lack context, while larger chunks offer more context but may introduce noise. The optimal 
    chunk size typically ranges from 500 to 2000 characters, depending on the domain and use case.
    """
    
    para3 = Paragraph(chapter3_content, styles['Normal'])
    story.append(para3)
    story.append(Spacer(1, 0.3*inch))
    
    # Chapter 4: Embedding Models
    chapter4_title = Paragraph("Chapter 4: Embedding Models and Representation Learning", styles['Heading1'])
    story.append(chapter4_title)
    story.append(Spacer(1, 0.2*inch))
    
    chapter4_content = """
    Embedding models are the foundation of semantic search in RAG systems. These models convert text into 
    dense vector representations that capture semantic meaning and enable similarity comparisons.
    
    Popular embedding models include:
    • all-MiniLM-L6-v2: A compact model with 384 dimensions, offering good performance-to-size ratio
    • all-mpnet-base-v2: Higher quality embeddings with 768 dimensions
    • text-embedding-ada-002: OpenAI's embedding model with 1536 dimensions
    • E5 models: Microsoft's multilingual embedding models
    
    Key considerations for embedding model selection:
    • Dimensionality: Higher dimensions generally capture more nuanced meanings but require more storage
    • Language support: Multilingual models for international applications
    • Domain specificity: Specialized models for technical, medical, or legal domains
    • Computational requirements: Balancing quality with inference speed
    
    The all-MiniLM-L6-v2 model, commonly used in RAG implementations, provides an excellent balance of 
    performance and efficiency. With 384 dimensions, it captures semantic relationships effectively while 
    maintaining reasonable storage and computational requirements.
    """
    
    para4 = Paragraph(chapter4_content, styles['Normal'])
    story.append(para4)
    story.append(Spacer(1, 0.3*inch))
    
    # Chapter 5: Implementation Best Practices
    chapter5_title = Paragraph("Chapter 5: RAG Implementation Best Practices", styles['Heading1'])
    story.append(chapter5_title)
    story.append(Spacer(1, 0.2*inch))
    
    chapter5_content = """
    Implementing a production-ready RAG system requires careful consideration of several factors:
    
    Architecture Design:
    • Modular design with clear separation of concerns
    • Scalable vector storage and retrieval systems
    • Efficient caching mechanisms for frequently accessed content
    • Robust error handling and fallback strategies
    
    Performance Optimization:
    • Batch processing for document ingestion
    • Asynchronous operations for improved responsiveness
    • Connection pooling for database operations
    • Intelligent caching of embeddings and search results
    
    Quality Assurance:
    • Comprehensive evaluation metrics (precision, recall, relevance)
    • A/B testing for different retrieval strategies
    • Human evaluation of generated responses
    • Continuous monitoring and improvement
    
    Security and Privacy:
    • Data encryption at rest and in transit
    • Access control and user isolation
    • Audit logging for compliance requirements
    • Privacy-preserving techniques for sensitive data
    
    The key to successful RAG implementation lies in balancing accuracy, performance, and maintainability 
    while ensuring the system can gracefully handle edge cases and failures.
    """
    
    para5 = Paragraph(chapter5_content, styles['Normal'])
    story.append(para5)
    
    # Build the PDF
    doc.build(story)
    print(f"✅ Created comprehensive test book: {filename}")
    return filename

if __name__ == "__main__":
    create_comprehensive_test_book()
