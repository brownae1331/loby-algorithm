import sys
import os

# Add project root to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from XGboost.manage2 import main_function  # Replace with your actual function

if __name__ == "__main__":
    main_function() 