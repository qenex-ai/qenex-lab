
import numpy as np
from decimal import Decimal
import os

# Add src to path
from interpreter import QValue, Dimensions

def test_vector_ops():
    print("Testing Decimal * Numpy Array interaction...")
    
    # Case 1: Decimal Scalar * Float Array
    d = Decimal("1.0000000000000000000000000000000000000000000000001")
    arr = np.array([1.0, 2.0, 3.0])
    
    q_scalar = QValue(d, Dimensions())
    q_vec = QValue(arr, Dimensions())
    
    try:
        # This calls q_scalar.__mul__(q_vec)
        res = q_scalar * q_vec
        print(f"Scalar * Vector Result: {res.value} (Type: {type(res.value)})")
        if isinstance(res.value, np.ndarray):
             print(f"Array Dtype: {res.value.dtype}")
    except Exception as e:
        print(f"FAILED Scalar * Vector: {e}")

    # Case 2: Vector * Decimal Scalar
    try:
        res2 = q_vec * q_scalar
        print(f"Vector * Scalar Result: {res2.value}")
    except Exception as e:
        print(f"FAILED Vector * Scalar: {e}")

if __name__ == "__main__":
    test_vector_ops()
