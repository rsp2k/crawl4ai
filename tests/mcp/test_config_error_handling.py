#!/usr/bin/env python3
"""
Tests for configuration error handling and validation
Tests invalid configurations and edge cases
"""
import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'deploy', 'docker'))

try:
    from schemas import MarkdownRequest, HTMLRequest, ScreenshotRequest
    from utils import FilterType
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    print("⚠️  Schemas not available - some tests will be skipped")


class TestConfigurationErrorHandling:
    """Test error handling for invalid configurations"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_invalid_filter_type(self):
        """Test invalid filter_type values are rejected"""
        with pytest.raises(ValueError):
            MarkdownRequest.model_validate({
                "url": "https://example.com",
                "filter_type": "invalid_filter"
            })
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_invalid_url_format(self):
        """Test invalid URL formats are rejected"""
        invalid_urls = [
            "not-a-url",
            "ftp://example.com",  # Wrong protocol
            "",                   # Empty string
            "just-text"          # No protocol
        ]
        
        for invalid_url in invalid_urls:
            with pytest.raises(ValueError):
                MarkdownRequest.model_validate({
                    "url": invalid_url,
                    "filter_type": "fit"
                })
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available") 
    def test_screenshot_invalid_wait_time(self):
        """Test screenshot wait time validation"""
        # Too short
        with pytest.raises(ValueError):
            ScreenshotRequest.model_validate({
                "url": "https://example.com",
                "screenshot_wait_for": 0.05  # Below minimum 0.1
            })
        
        # Too long
        with pytest.raises(ValueError):
            ScreenshotRequest.model_validate({
                "url": "https://example.com", 
                "screenshot_wait_for": 35.0  # Above maximum 30.0
            })
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_empty_scripts_array(self):
        """Test that empty scripts array is rejected"""
        from schemas import JSEndpointRequest
        
        with pytest.raises(ValueError):
            JSEndpointRequest.model_validate({
                "url": "https://example.com",
                "scripts": []  # Empty array should be rejected
            })


class TestConfigurationEdgeCases:
    """Test edge cases in configuration handling"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_extremely_large_configs(self):
        """Test handling of very large configuration objects"""
        # Create a large config with many properties
        large_browser_config = {
            f"property_{i}": f"value_{i}" for i in range(100)
        }
        large_crawler_config = {
            f"config_{i}": f"setting_{i}" for i in range(100)
        }
        
        # Should handle large configs without issue
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "filter_type": "fit",
            "browser_config": large_browser_config,
            "crawler_config": large_crawler_config
        })
        
        assert len(req.browser_config) == 100
        assert len(req.crawler_config) == 100
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_deeply_nested_configs(self):
        """Test deeply nested configuration structures"""
        deeply_nested_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "value": "deep_value"
                        }
                    }
                }
            }
        }
        
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "filter_type": "fit",
            "browser_config": deeply_nested_config
        })
        
        assert req.browser_config["level1"]["level2"]["level3"]["level4"]["value"] == "deep_value"
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_special_characters_in_configs(self):
        """Test special characters and unicode in configurations"""
        special_config = {
            "user_agent": "Mozilla/5.0 (特殊文字; Émojis 🚀; Symbols ©®™)",
            "custom_header": "Value with newlines\nand\ttabs",
            "unicode_value": "Iñtërnâtiônàlizætiøn",
            "symbols": "!@#$%^&*()_+-=[]{}|;:,.<>?"
        }
        
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "filter_type": "fit",
            "browser_config": special_config
        })
        
        assert "特殊文字" in req.browser_config["user_agent"]
        assert "🚀" in req.browser_config["user_agent"]
        assert "\n" in req.browser_config["custom_header"]
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_config_type_coercion(self):
        """Test that configs handle type coercion appropriately"""
        mixed_types_config = {
            "numeric_string": "123",
            "boolean_string": "true",
            "float_value": 3.14,
            "integer_value": 42,
            "boolean_value": False,
            "null_value": None
        }
        
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "filter_type": "fit",
            "browser_config": mixed_types_config
        })
        
        # Values should be preserved as-is in Dict type
        assert req.browser_config["numeric_string"] == "123"
        assert req.browser_config["boolean_string"] == "true"
        assert req.browser_config["float_value"] == 3.14
        assert req.browser_config["boolean_value"] is False


class TestBackwardsCompatibilityEdgeCases:
    """Test edge cases in backwards compatibility"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_mixed_old_and_new_parameters(self):
        """Test mixing old alias and new parameter names"""
        # This should work - new names take precedence
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "f": "raw",              # Old alias
            "filter_type": "fit",    # New name (should override)
            "q": "old query",        # Old alias
            "query": "new query"     # New name (should override)
        })
        
        # New names should take precedence
        assert req.filter_type.value == "fit"
        assert req.query == "new query"
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_empty_string_aliases(self):
        """Test empty string values in alias fields"""
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "f": "fit",
            "q": "",     # Empty string query
            "c": ""      # Empty string cache version
        })
        
        assert req.query == ""
        assert req.cache_version == ""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_none_values_in_aliases(self):
        """Test None values in alias fields"""
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "f": "fit",
            "q": None,   # Explicit None
            "c": None    # Explicit None
        })
        
        assert req.query is None
        assert req.cache_version is None


class TestConfigurationSerialization:
    """Test serialization and deserialization of configurations"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_json_roundtrip(self):
        """Test that configs survive JSON serialization roundtrip"""
        original_data = {
            "url": "https://example.com",
            "filter_type": "bm25",
            "query": "test query",
            "browser_config": {
                "headless": True,
                "viewport": {"width": 1920, "height": 1080}
            },
            "crawler_config": {
                "wait_for": "css:.content",
                "excluded_tags": ["nav", "footer"]
            }
        }
        
        # Create request
        req = MarkdownRequest.model_validate(original_data)
        
        # Serialize to JSON
        json_data = req.model_dump()
        
        # Deserialize back
        req2 = MarkdownRequest.model_validate(json_data)
        
        # Should be identical
        assert req2.url == req.url
        assert req2.filter_type == req.filter_type
        assert req2.browser_config == req.browser_config
        assert req2.crawler_config == req.crawler_config
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_json_with_aliases(self):
        """Test JSON serialization includes aliases when requested"""
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "filter_type": "fit",
            "query": "test",
            "cache_version": "1"
        })
        
        # Serialize with aliases
        json_with_aliases = req.model_dump(by_alias=True)
        
        # Should include alias names
        assert "f" in json_with_aliases
        assert "q" in json_with_aliases  
        assert "c" in json_with_aliases
        
        # Values should match
        assert json_with_aliases["f"] == "fit"
        assert json_with_aliases["q"] == "test"
        assert json_with_aliases["c"] == "1"


def run_error_handling_tests():
    """Run all error handling tests manually"""
    import unittest
    
    test_classes = [
        TestConfigurationErrorHandling,
        TestConfigurationEdgeCases, 
        TestBackwardsCompatibilityEdgeCases,
        TestConfigurationSerialization
    ]
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n🛡️  Error Handling Tests Summary:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n💥 Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_error_handling_tests()
    if not success:
        sys.exit(1)
    print("🎉 All error handling tests passed!")