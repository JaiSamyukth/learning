#!/usr/bin/env python3
"""
Test RAG system with API key rotation
"""

import requests
import json

def test_chat():
    """Test chat functionality with API key rotation"""
    print("🔄 Testing RAG Chat with API Key Rotation...")
    
    try:
        response = requests.post('http://localhost:8000/api/chat/', 
                               json={'message': 'What are the key components of a RAG system?'},
                               timeout=30)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', 'No response field')
            print(f"✅ Success! Response: {response_text[:200]}...")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

def test_question_generation():
    """Test question generation with API key rotation"""
    print("\n🔄 Testing Question Generation with API Key Rotation...")
    
    try:
        response = requests.post('http://localhost:8000/api/chat/generate-questions', 
                               json={'topic': '', 'count': 3, 'mode': 'practice'},
                               timeout=45)
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            response_text = data.get('response', 'No response field')
            print(f"✅ Success! Generated questions: {response_text[:200]}...")
            return True
        else:
            print(f"❌ Error: {response.text}")
            return False
            
    except requests.exceptions.Timeout:
        print("⏰ Request timed out")
        return False
    except Exception as e:
        print(f"❌ Exception: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Testing RAG System with API Key Rotation")
    print("=" * 50)
    
    # Test chat
    chat_success = test_chat()
    
    # Test question generation
    questions_success = test_question_generation()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Chat: {'✅ PASS' if chat_success else '❌ FAIL'}")
    print(f"Questions: {'✅ PASS' if questions_success else '❌ FAIL'}")
    
    if chat_success and questions_success:
        print("\n🎉 ALL TESTS PASSED! RAG system with API key rotation is working!")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")
