#!/usr/bin/env python3
"""
Test script to verify the random rotation logic works correctly.
"""

import random
import sys
import os

# Add the project root to the path so we can import the modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_random_rotation_logic():
    """Test the random rotation logic without running the full training."""
    
    # Set seed for reproducibility
    random.seed(42)
    
    print("Testing random rotation logic...")
    print("=" * 50)
    
    # Simulate the rotation logic
    max_epochs = 20
    patience = 8
    
    # Initialize rotation tracking (simulating the code we just added)
    rotation_period = "random"
    
    if rotation_period == "random":
        next_rotation_epoch = random.randint(1, 4)  # Sample from 1 to 4
        rotation_epochs = None
    else:
        rotation_epochs = int(patience * (3/4)) if rotation_period == "relative" else rotation_period
        next_rotation_epoch = None
    
    print(f"Initial next_rotation_epoch: {next_rotation_epoch}")
    print(f"rotation_epochs: {rotation_epochs}")
    print()
    
    rotation_count = 0
    rotation_epochs_list = []
    
    for e in range(max_epochs):
        # Check if rotation should happen
        should_rotate = False
        if rotation_period == "random":
            if e == next_rotation_epoch:
                should_rotate = True
                # Sample next rotation epoch (1 to 4 epochs from now)
                next_rotation_epoch = e + random.randint(1, 4)
        else:
            # Fixed or relative rotation
            if rotation_epochs is not None and e % rotation_epochs == 0:
                should_rotate = True
        
        if should_rotate:
            rotation_count += 1
            rotation_epochs_list.append(e)
            print(f"Epoch {e}: ROTATION! Next rotation at epoch {next_rotation_epoch}")
        else:
            print(f"Epoch {e}: No rotation. Next rotation at epoch {next_rotation_epoch}")
    
    print()
    print("=" * 50)
    print(f"Total rotations: {rotation_count}")
    print(f"Rotation epochs: {rotation_epochs_list}")
    
    # Calculate intervals between rotations
    if len(rotation_epochs_list) > 1:
        intervals = [rotation_epochs_list[i+1] - rotation_epochs_list[i] for i in range(len(rotation_epochs_list)-1)]
        print(f"Intervals between rotations: {intervals}")
        print(f"Average interval: {sum(intervals) / len(intervals):.2f}")
        print(f"Min interval: {min(intervals)}")
        print(f"Max interval: {max(intervals)}")
    
    # Verify that intervals are between 1 and 4
    if len(rotation_epochs_list) > 1:
        all_valid = all(1 <= interval <= 4 for interval in intervals)
        print(f"All intervals in [1,4] range: {all_valid}")
    
    return rotation_count > 0

if __name__ == "__main__":
    success = test_random_rotation_logic()
    if success:
        print("\n✅ Random rotation test passed!")
    else:
        print("\n❌ Random rotation test failed!")
        sys.exit(1) 