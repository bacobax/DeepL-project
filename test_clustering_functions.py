#!/usr/bin/env python3
"""
Test script for the new clustering functions.
"""

from utils.clustering import rotate_clusters, clusters_history

def test_rotate_clusters():
    """Test the rotate_clusters function."""
    print("Testing rotate_clusters function...")
    
    # Test with different split ratios
    split_ratios = [0.7, 0.8, 0.9]
    
    for split_ratio in split_ratios:
        print(f"\nTesting with split_ratio = {split_ratio}")
        cluster_assignments = rotate_clusters(split_ratio)
        
        # Count categories in each cluster
        cluster_0_count = sum(1 for v in cluster_assignments.values() if v == 0)
        cluster_1_count = sum(1 for v in cluster_assignments.values() if v == 1)
        total_categories = len(cluster_assignments)
        
        print(f"Total categories: {total_categories}")
        print(f"Categories in cluster 0: {cluster_0_count}")
        print(f"Categories in cluster 1: {cluster_1_count}")
        print(f"Expected cluster 0 ratio: {split_ratio}")
        print(f"Actual cluster 0 ratio: {cluster_0_count / total_categories:.3f}")
        
        # Verify the split ratio is approximately correct
        expected_cluster_0 = int(split_ratio * total_categories)
        assert cluster_0_count == expected_cluster_0, f"Expected {expected_cluster_0} categories in cluster 0, got {cluster_0_count}"
        print("✅ Test passed!")

def test_clusters_history():
    """Test the clusters_history function."""
    print("\nTesting clusters_history function...")
    
    # Test with a smaller split ratio to avoid too many combinations
    split_ratio = 0.7
    print(f"Testing with split_ratio = {split_ratio}")
    
    cluster_configurations = clusters_history(split_ratio)
    
    print(f"Generated {len(cluster_configurations)} cluster configurations")
    
    # Test a few configurations
    for i, config in enumerate(cluster_configurations[:3]):  # Test first 3 configurations
        cluster_0_count = sum(1 for v in config.values() if v == 0)
        cluster_1_count = sum(1 for v in config.values() if v == 1)
        total_categories = len(config)
        
        print(f"\nConfiguration {i+1}:")
        print(f"  Total categories: {total_categories}")
        print(f"  Categories in cluster 0: {cluster_0_count}")
        print(f"  Categories in cluster 1: {cluster_1_count}")
        print(f"  Cluster 0 ratio: {cluster_0_count / total_categories:.3f}")
        
        # Verify the split ratio is correct
        expected_cluster_0 = int(split_ratio * total_categories)
        assert cluster_0_count == expected_cluster_0, f"Expected {expected_cluster_0} categories in cluster 0, got {cluster_0_count}"
    
    print("✅ Test passed!")

if __name__ == "__main__":
    test_rotate_clusters()
    test_clusters_history()
    print("\n🎉 All tests passed!") 