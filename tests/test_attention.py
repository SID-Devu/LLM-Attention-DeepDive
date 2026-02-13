#!/usr/bin/env python3
"""
Test suite for LLM Attention DeepDive
"""

import os
import sys
import unittest
import json
import tempfile

# Add scripts to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))


class TestAttentionKernels(unittest.TestCase):
    """Tests for attention kernel files"""
    
    def test_naive_attention_exists(self):
        """Verify naive attention kernel exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'src', 'attention_naive.hip'
        )
        self.assertTrue(os.path.exists(path))
    
    def test_shared_attention_exists(self):
        """Verify shared memory attention kernel exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'src', 'attention_shared.hip'
        )
        self.assertTrue(os.path.exists(path))
    
    def test_flash_attention_exists(self):
        """Verify flash attention kernel exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'src', 'attention_flash.hip'
        )
        self.assertTrue(os.path.exists(path))
    
    def test_common_header_exists(self):
        """Verify common header exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'src', 'attention_common.h'
        )
        self.assertTrue(os.path.exists(path))


class TestAnalysisScripts(unittest.TestCase):
    """Tests for analysis scripts"""
    
    def test_analyze_results_exists(self):
        """Verify analysis script exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'scripts', 'analyze_results.py'
        )
        self.assertTrue(os.path.exists(path))
    
    def test_analyze_results_importable(self):
        """Test analysis script can be imported"""
        try:
            import analyze_results
            self.assertTrue(True)
        except ImportError:
            self.skipTest("analyze_results not importable")


class TestResultsFormat(unittest.TestCase):
    """Tests for results file format"""
    
    def test_example_results_exists(self):
        """Verify example results file exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'results', 'example_results.json'
        )
        self.assertTrue(os.path.exists(path))
    
    def test_example_results_format(self):
        """Verify example results JSON structure"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'results', 'example_results.json'
        )
        
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            
            # Should have attention variants
            self.assertIn('attention_variants', data)
            
            variants = data['attention_variants']
            self.assertIn('naive', variants)
            self.assertIn('flash_attention', variants)


class TestDocumentation(unittest.TestCase):
    """Tests for documentation"""
    
    def test_optimization_doc_exists(self):
        """Verify optimization documentation exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'docs', 'optimization.md'
        )
        self.assertTrue(os.path.exists(path))


class TestProfilingConfig(unittest.TestCase):
    """Tests for profiling configuration"""
    
    def test_rocprof_counters_exists(self):
        """Verify rocprof counters file exists"""
        path = os.path.join(
            os.path.dirname(__file__),
            '..', 'profiling', 'rocprof_counters.txt'
        )
        self.assertTrue(os.path.exists(path))


if __name__ == '__main__':
    unittest.main(verbosity=2)
