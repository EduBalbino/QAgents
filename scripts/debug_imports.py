print("--- Starting debug script ---")
print("Python interpreter:", __import__('sys').executable)

try:
    print("Importing sys...")
    import sys
    print("Importing os...")
    import os
    print("Importing datetime...")
    import datetime
    print("--- Basic imports successful ---")

    print("Importing pennylane...")
    import pennylane as qml
    print("Importing pennylane.numpy...")
    from pennylane import numpy as np
    print("--- Pennylane successful ---")

    print("Importing pandas...")
    import pandas as pd
    print("--- Pandas successful ---")

    print("Importing sklearn...")
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    print("--- Sklearn successful ---")

    print("Importing time...")
    import time
    print("Importing random...")
    import random
    print("--- Utility imports successful ---")

    print("\n✅ ALL IMPORTS WERE SUCCESSFUL!")

except Exception as e:
    print(f"\n❌ SCRIPT FAILED ON IMPORT with error: {e}")
    # Also print the traceback
    import traceback
    traceback.print_exc()
