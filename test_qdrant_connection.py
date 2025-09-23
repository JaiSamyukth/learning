#!/usr/bin/env python3
"""
Qdrant Cloud Connection Test
Tests the Qdrant Cloud connection with provided credentials
"""

import asyncio
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

from backend.services.vector_service import vector_service
from backend.config.qdrant_config import qdrant_config


async def test_qdrant_connection():
    """Test Qdrant Cloud connection"""
    print("🔄 Testing Qdrant Cloud Connection...")
    print(f"📍 URL: {qdrant_config.QDRANT_URL}")
    print(f"🔑 API Key: {qdrant_config.QDRANT_API_KEY[:20]}...")

    try:
        # Test connection
        health = vector_service.health_check()

        if health.get('status') == 'healthy':
            print("✅ Qdrant Cloud connection successful!")
            print(f"📊 Status: {health}")

            # Test collection operations
            print("\n🔍 Testing collection operations...")
            collections = vector_service._qdrant_client.get_collections()
            print(f"📚 Available collections: {[c.name for c in collections.collections]}")

            # Test embedding model
            print("\n🧠 Testing embedding model...")
            test_embedding = vector_service._embedding_model.encode("test query")
            print(f"✅ Embedding model working! Vector dimension: {len(test_embedding)}")

            print("\n🎉 All tests passed! Qdrant Cloud integration is ready.")
            return True
        else:
            print(f"❌ Qdrant health check failed: {health}")
            return False

    except Exception as e:
        print(f"❌ Qdrant connection test failed: {str(e)}")
        return False


async def test_database_connection():
    """Test database connection"""
    print("\n🔄 Testing Database Connection...")

    try:
        from backend.services.database_service import database_service

        health = database_service.health_check()

        if health.get('status') == 'healthy':
            print("✅ Database connection successful!")
            print(f"📊 Status: {health}")
            return True
        else:
            print(f"❌ Database health check failed: {health}")
            return False

    except Exception as e:
        print(f"❌ Database connection test failed: {str(e)}")
        return False


async def main():
    """Run all connection tests"""
    print("🚀 Lumina IQ RAG System - Connection Tests")
    print("=" * 50)

    # Test Qdrant connection
    qdrant_ok = await test_qdrant_connection()

    # Test database connection
    db_ok = await test_database_connection()

    print("\n" + "=" * 50)
    if qdrant_ok and db_ok:
        print("🎉 All connection tests passed!")
        print("✅ Lumina IQ RAG system is ready for deployment")
        return 0
    else:
        print("❌ Some connection tests failed")
        print("🔧 Please check your configuration and try again")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)