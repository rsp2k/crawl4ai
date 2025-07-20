#!/usr/bin/env python3
"""
Tests for enhanced MCP configuration support
Tests browser_config and crawler_config parameters across all tools
"""
import pytest
import sys
import os
import asyncio
from unittest.mock import patch, MagicMock

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'deploy', 'docker'))

from schemas import (
    MarkdownRequest, HTMLRequest, ScreenshotRequest, 
    PDFRequest, JSEndpointRequest, CrawlRequest
)


class TestEnhancedConfigSchemas:
    """Test that all enhanced schemas accept configuration parameters"""
    
    def test_markdown_request_with_configs(self):
        """Test MarkdownRequest accepts browser_config and crawler_config"""
        request_data = {
            "url": "https://example.com",
            "filter_type": "fit",
            "query": "test query",
            "cache_version": "1",
            "browser_config": {
                "headless": False,
                "viewport": {"width": 1920, "height": 1080},
                "user_agent": "Mozilla/5.0 Test Agent"
            },
            "crawler_config": {
                "wait_for": "css:.main-content",
                "excluded_tags": ["nav", "footer"],
                "cache_mode": "bypass",
                "page_timeout": 30000
            }
        }
        
        # Should not raise any validation errors
        req = MarkdownRequest.model_validate(request_data)
        
        # Verify all fields are populated correctly
        assert req.url == "https://example.com"
        assert req.filter_type.value == "fit"
        assert req.query == "test query"
        assert req.cache_version == "1"
        assert req.browser_config["headless"] is False
        assert req.browser_config["viewport"]["width"] == 1920
        assert req.crawler_config["wait_for"] == "css:.main-content"
        assert "nav" in req.crawler_config["excluded_tags"]
    
    def test_markdown_request_backwards_compatibility(self):
        """Test MarkdownRequest still accepts old parameter names"""
        request_data = {
            "url": "https://example.com",
            "f": "fit",           # Old alias
            "q": "test query",    # Old alias  
            "c": "1"              # Old alias
        }
        
        req = MarkdownRequest.model_validate(request_data)
        
        assert req.filter_type.value == "fit"
        assert req.query == "test query"
        assert req.cache_version == "1"
        # Config fields should have defaults
        assert req.browser_config == {}
        assert req.crawler_config == {}
    
    def test_html_request_with_configs(self):
        """Test HTMLRequest accepts configuration parameters"""
        request_data = {
            "url": "https://example.com",
            "browser_config": {
                "headless": True,
                "proxy": "http://proxy:8080"
            },
            "crawler_config": {
                "wait_for": "css:.content-loaded",
                "only_text": True
            }
        }
        
        req = HTMLRequest.model_validate(request_data)
        assert req.browser_config["headless"] is True
        assert req.browser_config["proxy"] == "http://proxy:8080"
        assert req.crawler_config["wait_for"] == "css:.content-loaded"
        assert req.crawler_config["only_text"] is True
    
    def test_screenshot_request_with_configs(self):
        """Test ScreenshotRequest accepts configuration parameters"""
        request_data = {
            "url": "https://mobile-site.com",
            "screenshot_wait_for": 3.0,
            "output_path": "/tmp/screenshot.png",
            "browser_config": {
                "viewport": {"width": 375, "height": 667},
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
            },
            "crawler_config": {
                "wait_for": "css:.mobile-ready",
                "page_timeout": 45000
            }
        }
        
        req = ScreenshotRequest.model_validate(request_data)
        assert req.screenshot_wait_for == 3.0
        assert req.browser_config["viewport"]["width"] == 375
        assert req.browser_config["user_agent"].startswith("Mozilla/5.0 (iPhone")
        assert req.crawler_config["wait_for"] == "css:.mobile-ready"
    
    def test_pdf_request_with_configs(self):
        """Test PDFRequest accepts configuration parameters"""
        request_data = {
            "url": "https://docs-site.com",
            "output_path": "/tmp/document.pdf",
            "browser_config": {
                "headless": True,
                "viewport": {"width": 1200, "height": 800}
            },
            "crawler_config": {
                "excluded_tags": ["script", "style", "nav"],
                "wait_for": "css:.doc-content"
            }
        }
        
        req = PDFRequest.model_validate(request_data)
        assert req.browser_config["viewport"]["width"] == 1200
        assert "script" in req.crawler_config["excluded_tags"]
        assert req.crawler_config["wait_for"] == "css:.doc-content"
    
    def test_js_request_with_configs(self):
        """Test JSEndpointRequest accepts configuration parameters"""
        request_data = {
            "url": "https://dynamic-app.com",
            "scripts": ["document.title", "window.location.href"],
            "browser_config": {
                "headless": False,
                "java_script_enabled": True,
                "viewport": {"width": 1366, "height": 768}
            },
            "crawler_config": {
                "session_id": "test-session",
                "wait_for": "css:.app-ready",
                "page_timeout": 60000
            }
        }
        
        req = JSEndpointRequest.model_validate(request_data)
        assert req.scripts == ["document.title", "window.location.href"]
        assert req.browser_config["java_script_enabled"] is True
        assert req.crawler_config["session_id"] == "test-session"
    
    def test_crawl_request_already_has_configs(self):
        """Test CrawlRequest maintains its existing config support"""
        request_data = {
            "urls": ["https://site1.com", "https://site2.com"],
            "browser_config": {"headless": True},
            "crawler_config": {"cache_mode": "bypass"}
        }
        
        req = CrawlRequest.model_validate(request_data)
        assert len(req.urls) == 2
        assert req.browser_config["headless"] is True
        assert req.crawler_config["cache_mode"] == "bypass"


class TestConfigurationValidation:
    """Test validation of configuration parameters"""
    
    def test_empty_configs_are_valid(self):
        """Test that empty configs default to empty dictionaries"""
        req = MarkdownRequest.model_validate({
            "url": "https://example.com",
            "filter_type": "fit"
        })
        
        assert req.browser_config == {}
        assert req.crawler_config == {}
    
    def test_none_configs_default_to_empty(self):
        """Test that None configs are converted to empty dicts"""
        req = MarkdownRequest.model_validate({
            "url": "https://example.com", 
            "filter_type": "fit",
            "browser_config": None,
            "crawler_config": None
        })
        
        assert req.browser_config == {}
        assert req.crawler_config == {}
    
    def test_complex_nested_configs(self):
        """Test complex nested configuration structures"""
        request_data = {
            "url": "https://complex-site.com",
            "filter_type": "bm25",
            "query": "advanced search",
            "browser_config": {
                "viewport": {
                    "width": 1920,
                    "height": 1080,
                    "deviceScaleFactor": 2
                },
                "proxy": {
                    "server": "http://proxy.example.com:8080",
                    "username": "user",
                    "password": "pass"
                }
            },
            "crawler_config": {
                "excluded_tags": ["script", "style", "noscript"],
                "wait_for": "css:.content-ready",
                "extraction_strategy": "LLMExtractionStrategy",
                "word_count_threshold": 15
            }
        }
        
        req = MarkdownRequest.model_validate(request_data)
        
        # Verify nested structures
        assert req.browser_config["viewport"]["deviceScaleFactor"] == 2
        assert req.browser_config["proxy"]["server"] == "http://proxy.example.com:8080"
        assert req.crawler_config["word_count_threshold"] == 15
        assert len(req.crawler_config["excluded_tags"]) == 3


class TestSchemaGeneration:
    """Test that enhanced schemas generate proper JSON schemas"""
    
    def test_md_schema_includes_config_fields(self):
        """Test MarkdownRequest schema includes config fields with examples"""
        schema = MarkdownRequest.model_json_schema()
        properties = schema["properties"]
        
        # Check config fields exist
        assert "browser_config" in properties
        assert "crawler_config" in properties
        
        # Check they have proper descriptions
        assert "Browser configuration options" in properties["browser_config"]["description"]
        assert "Crawler behavior configuration" in properties["crawler_config"]["description"]
        
        # Check they have examples
        assert "examples" in properties["browser_config"]
        assert "examples" in properties["crawler_config"]
        assert len(properties["browser_config"]["examples"]) > 0
        assert len(properties["crawler_config"]["examples"]) > 0
    
    def test_all_schemas_have_config_fields(self):
        """Test all enhanced schemas include config fields"""
        schemas_to_test = [
            (MarkdownRequest, "md"),
            (HTMLRequest, "html"),
            (ScreenshotRequest, "screenshot"),
            (PDFRequest, "pdf"),
            (JSEndpointRequest, "execute_js")
        ]
        
        for schema_class, tool_name in schemas_to_test:
            schema = schema_class.model_json_schema()
            properties = schema["properties"]
            
            # All should have config fields
            assert "browser_config" in properties, f"{tool_name} missing browser_config"
            assert "crawler_config" in properties, f"{tool_name} missing crawler_config"
            
            # All should have examples
            browser_examples = properties["browser_config"].get("examples", [])
            crawler_examples = properties["crawler_config"].get("examples", [])
            
            assert len(browser_examples) > 0, f"{tool_name} browser_config has no examples"
            assert len(crawler_examples) > 0, f"{tool_name} crawler_config has no examples"


if __name__ == "__main__":
    # Run tests directly
    import unittest
    
    # Convert pytest classes to unittest
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Add test classes
    for test_class in [TestEnhancedConfigSchemas, TestConfigurationValidation, TestSchemaGeneration]:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n🎯 Enhanced Config Tests Summary:")
    print(f"✅ Tests run: {result.testsRun}")
    print(f"❌ Failures: {len(result.failures)}")
    print(f"⚠️  Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("🎉 All enhanced configuration tests passed!")
    else:
        print("💥 Some tests failed - check output above")
        sys.exit(1)