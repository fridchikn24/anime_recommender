import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

print("Python path:", sys.path)

try:
    from src.recommender.store import ChromaStore
    print("ChromaStore imported successfully")
except Exception as e:
    print("Import failed:", e)

# List all names in the module
import src.recommender.store as mod
print("Module names:", dir(mod))
