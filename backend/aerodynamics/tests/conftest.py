import sys, os

# Add the parent of "aerodynamics/" (i.e. "backend/") onto PYTHONPATH
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
