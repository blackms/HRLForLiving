"""Test script to verify all imports work correctly"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Testing imports...")

try:
    print("✓ Importing training_service...")
    from backend.services.training_service import TrainingService, training_service
    print("  - TrainingService class imported")
    print("  - training_service instance imported")
    
    print("✓ Importing training_socket...")
    from backend.websocket.training_socket import sio, socket_manager
    print("  - sio instance imported")
    print("  - socket_manager instance imported")
    
    print("✓ Importing training API...")
    from backend.api.training import router
    print("  - training router imported")
    
    print("\n✅ All imports successful!")
    print("\nNote: To run the server, install dependencies:")
    print("  pip install -r backend/requirements.txt")
    print("\nThen start the server:")
    print("  uvicorn backend.main:socket_app --reload --port 8000")
    
except ImportError as e:
    print(f"\n❌ Import error: {e}")
    print("\nPlease install dependencies:")
    print("  pip install -r backend/requirements.txt")
    sys.exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
