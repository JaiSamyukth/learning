#!/usr/bin/env python3
"""
Test vector service with API key rotation
"""

import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

from services.vector_service import vector_service

async def test_search():
    """Test vector search with API key rotation"""
    print('🔄 Testing vector service search with API key rotation...')
    try:
        # Test a simple search
        result = await vector_service.search_similar('test query', 'test_user', limit=1)
        print(f'✅ Search successful: {len(result)} results')
        return True
    except Exception as e:
        print(f'❌ Search failed: {e}')
        return False

async def test_health():
    """Test vector service health check"""
    print('🔄 Testing vector service health check...')
    try:
        health = vector_service.health_check()
        print(f'✅ Health check successful: {health["status"]}')
        print(f'   Embedding model: {health.get("embedding_model", "unknown")}')
        print(f'   Dimensions: {health.get("embedding_dimensions", "unknown")}')
        return True
    except Exception as e:
        print(f'❌ Health check failed: {e}')
        return False

async def main():
    """Main test function"""
    print("🚀 Testing Vector Service with API Key Rotation")
    print("=" * 50)
    
    # Test health check
    health_success = await test_health()
    
    # Test search
    search_success = await test_search()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"Health: {'✅ PASS' if health_success else '❌ FAIL'}")
    print(f"Search: {'✅ PASS' if search_success else '❌ FAIL'}")
    
    if health_success and search_success:
        print("\n🎉 ALL TESTS PASSED! Vector service with API key rotation is working!")
    else:
        print("\n⚠️  Some tests failed. Check the logs for details.")

if __name__ == "__main__":
    asyncio.run(main())
