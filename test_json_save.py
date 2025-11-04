#!/usr/bin/env python3
"""Test JSON saving with NaN values"""

import json
import numpy as np

# Test different types of NaN
values = [
    1.0,
    np.float32(1.0),
    np.float64(1.0),
    float('nan'),
    np.nan,
    np.float32('nan'),
    np.float64('nan'),
]

print("Testing JSON serialization of different types:")
for v in values:
    print(f"Value: {v}, Type: {type(v)}, Is NaN: {np.isnan(v) if isinstance(v, (float, np.floating)) else 'N/A'}")
    try:
        json_str = json.dumps({'value': float(v)})
        print(f"  JSON: {json_str}")
    except Exception as e:
        print(f"  Error: {e}")

# Test with list
print("\nTesting list conversion:")
test_list = [1.0, 2.0, np.nan, 4.0]
converted = [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in test_list]
print(f"Original: {test_list}")
print(f"Converted: {converted}")
print(f"JSON: {json.dumps(converted)}")
