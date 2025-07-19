#!/usr/bin/env python3
"""
Tests to verify enhanced configs work with existing test patterns
Ensures backwards compatibility and new functionality coexist
"""
import pytest
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'deploy', 'docker'))

try:
    from schemas import MarkdownRequest, HTMLRequest, ScreenshotRequest
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    print("⚠️  Schemas not available - compatibility tests will be skipped")


class TestExistingPatternCompatibility:
    """Test that existing test patterns still work with enhanced schemas"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_existing_md_pattern_still_works(self):
        """Test pattern from existing test_mcp_socket.py"""
        # This is the exact pattern from the existing test
        existing_pattern = {
            "url": "https://example.com",
            "f": "fit",   # Old alias
            "q": None,    # Old alias
            "c": "0",     # Old alias
        }
        
        # Should work exactly as before
        req = MarkdownRequest.model_validate(existing_pattern)
        
        assert req.url == "https://example.com"
        assert req.filter_type.value == "fit"
        assert req.query is None
        assert req.cache_version == "0"
        
        # New config fields should have defaults
        assert req.browser_config == {}
        assert req.crawler_config == {}
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_simple_api_test_pattern(self):
        """Test pattern from simple_api_test.py"""
        # Pattern from existing simple_api_test.py
        simple_pattern = {
            "url": "https://example.com",
            "f": "fit",
            "q": "test query",
            "c": "0"
        }
        
        req = MarkdownRequest.model_validate(simple_pattern)
        
        assert req.filter_type.value == "fit"
        assert req.query == "test query"
        assert req.cache_version == "0"
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_html_minimal_pattern(self):
        """Test minimal HTML request pattern"""
        minimal_html = {"url": "https://example.com"}
        
        req = HTMLRequest.model_validate(minimal_html)
        
        assert req.url == "https://example.com"
        assert req.browser_config == {}
        assert req.crawler_config == {}
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_screenshot_basic_pattern(self):
        """Test basic screenshot pattern"""
        basic_screenshot = {
            "url": "https://example.com",
            "screenshot_wait_for": 1.0,
        }
        
        req = ScreenshotRequest.model_validate(basic_screenshot)
        
        assert req.url == "https://example.com"
        assert req.screenshot_wait_for == 1.0
        assert req.browser_config == {}
        assert req.crawler_config == {}


class TestEnhancedCompatibility:
    """Test that enhanced features work alongside existing patterns"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_gradual_enhancement_md(self):
        """Test gradually adding enhanced features to existing pattern"""
        # Start with existing pattern
        base_request = {
            "url": "https://example.com",
            "f": "fit",
            "q": "test query",
            "c": "0"
        }
        
        # Add browser config only
        with_browser_config = base_request.copy()
        with_browser_config["browser_config"] = {"headless": False}
        
        req1 = MarkdownRequest.model_validate(with_browser_config)
        assert req1.browser_config["headless"] is False
        assert req1.crawler_config == {}
        
        # Add crawler config too
        with_both_configs = with_browser_config.copy()
        with_both_configs["crawler_config"] = {"wait_for": "css:.content"}
        
        req2 = MarkdownRequest.model_validate(with_both_configs)
        assert req2.browser_config["headless"] is False
        assert req2.crawler_config["wait_for"] == "css:.content"
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_mixed_old_new_parameters(self):
        """Test mixing old aliases with new config parameters"""
        mixed_request = {
            "url": "https://example.com",
            "f": "bm25",                    # Old alias
            "query": "search term",        # New name (should override q if present)
            "c": "1",                      # Old alias
            "browser_config": {            # New config
                "headless": True,
                "viewport": {"width": 1920, "height": 1080}
            },
            "crawler_config": {            # New config
                "wait_for": "css:.loaded",
                "excluded_tags": ["nav", "footer"]
            }
        }
        
        req = MarkdownRequest.model_validate(mixed_request)
        
        # Old aliases should work
        assert req.filter_type.value == "bm25"
        assert req.cache_version == "1"
        
        # New name should work
        assert req.query == "search term"
        
        # New configs should work
        assert req.browser_config["headless"] is True
        assert req.browser_config["viewport"]["width"] == 1920
        assert req.crawler_config["wait_for"] == "css:.loaded"
        assert "nav" in req.crawler_config["excluded_tags"]
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_json_schema_backwards_compatibility(self):
        """Test that JSON schema still includes old aliases"""
        schema = MarkdownRequest.model_json_schema()
        properties = schema["properties"]
        
        # Should include old aliases for backwards compatibility
        assert "f" in properties
        assert "q" in properties
        assert "c" in properties
        
        # Should also include new config fields
        assert "browser_config" in properties
        assert "crawler_config" in properties
        
        # Old aliases should reference new field names
        # (This is handled by Pydantic's alias system)


class TestResponseCompatibility:
    """Test that enhanced schemas produce compatible responses"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_serialization_with_aliases(self):
        """Test serialization maintains compatibility"""
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "filter_type": "fit",
            "query": "test",
            "cache_version": "1",
            "browser_config": {"headless": True},
            "crawler_config": {"wait_for": "css:.content"}
        })
        
        # Serialize with aliases (for backwards compatibility)
        json_with_aliases = req.model_dump(by_alias=True)
        
        # Should include old alias names
        assert "f" in json_with_aliases
        assert "q" in json_with_aliases
        assert "c" in json_with_aliases
        
        # Should include new config fields
        assert "browser_config" in json_with_aliases
        assert "crawler_config" in json_with_aliases
        
        # Values should be correct
        assert json_with_aliases["f"] == "fit"
        assert json_with_aliases["q"] == "test"
        assert json_with_aliases["c"] == "1"
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_serialization_without_aliases(self):
        """Test serialization with new field names"""
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "f": "fit",     # Using old aliases
            "q": "test",
            "c": "1"
        })
        
        # Serialize without aliases (new format)
        json_without_aliases = req.model_dump(by_alias=False)
        
        # Should use new field names
        assert "filter_type" in json_without_aliases
        assert "query" in json_without_aliases
        assert "cache_version" in json_without_aliases
        
        # Should NOT include aliases
        assert "f" not in json_without_aliases
        assert "q" not in json_without_aliases  
        assert "c" not in json_without_aliases


class TestSchemaEvolution:
    """Test that schemas can evolve while maintaining compatibility"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_future_config_additions(self):
        """Test that arbitrary config additions are accepted"""
        # Simulate future config options that don't exist yet
        future_config = {
            "url": "https://example.com",
            "filter_type": "fit",
            "browser_config": {
                "future_option_1": "value1",
                "experimental_feature": True,
                "new_viewport_setting": {"zoom": 1.5}
            },
            "crawler_config": {
                "future_extraction_mode": "advanced",
                "ai_enhancement": True,
                "custom_processor": {"type": "experimental"}
            }
        }
        
        req = MarkdownRequest.model_validate(future_config)
        
        # Future options should be preserved
        assert req.browser_config["future_option_1"] == "value1"
        assert req.browser_config["experimental_feature"] is True
        assert req.crawler_config["ai_enhancement"] is True
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_config_field_flexibility(self):
        """Test that config fields accept various data types"""
        flexible_config = {
            "url": "https://example.com",
            "filter_type": "fit",
            "browser_config": {
                "string_value": "text",
                "number_value": 42,
                "float_value": 3.14,
                "boolean_value": True,
                "null_value": None,
                "array_value": [1, 2, 3],
                "object_value": {"nested": "data"}
            }
        }
        
        req = MarkdownRequest.model_validate(flexible_config)
        
        # All types should be preserved
        assert req.browser_config["string_value"] == "text"
        assert req.browser_config["number_value"] == 42
        assert req.browser_config["float_value"] == 3.14
        assert req.browser_config["boolean_value"] is True
        assert req.browser_config["null_value"] is None
        assert req.browser_config["array_value"] == [1, 2, 3]
        assert req.browser_config["object_value"]["nested"] == "data"


def run_compatibility_tests():
    """Run all compatibility tests manually"""
    import unittest
    
    test_classes = [
        TestExistingPatternCompatibility,
        TestEnhancedCompatibility,
        TestResponseCompatibility,
        TestSchemaEvolution
    ]
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n🔄 Compatibility Tests Summary:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n💥 Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_compatibility_tests()
    if not success:
        sys.exit(1)
    print("🎉 All compatibility tests passed!")