#!/usr/bin/env python3
"""
Tests for real-world configuration scenarios
Tests practical use cases that LLMs might encounter
"""
import pytest
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'deploy', 'docker'))

try:
    from schemas import (
        MarkdownRequest, HTMLRequest, ScreenshotRequest, 
        PDFRequest, JSEndpointRequest
    )
    SCHEMAS_AVAILABLE = True
except ImportError:
    SCHEMAS_AVAILABLE = False
    print("⚠️  Schemas not available - real-world scenario tests will be skipped")


class TestMobileWebScenarios:
    """Test mobile web crawling scenarios"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_mobile_screenshot_configuration(self):
        """Test mobile screenshot with iPhone simulation"""
        mobile_config = {
            "url": "https://responsive-website.com",
            "screenshot_wait_for": 3.0,
            "output_path": "/tmp/mobile_screenshot.png",
            "browser_config": {
                "headless": True,
                "viewport": {
                    "width": 375,
                    "height": 667,
                    "deviceScaleFactor": 2,
                    "isMobile": True
                },
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
            },
            "crawler_config": {
                "wait_for": "css:.mobile-content-loaded",
                "page_timeout": 45000,
                "excluded_tags": ["script", "noscript"]
            }
        }
        
        req = ScreenshotRequest.model_validate(mobile_config)
        
        # Verify mobile-specific settings
        assert req.browser_config["viewport"]["width"] == 375
        assert req.browser_config["viewport"]["isMobile"] is True
        assert "iPhone" in req.browser_config["user_agent"]
        assert req.crawler_config["wait_for"] == "css:.mobile-content-loaded"
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_tablet_markdown_extraction(self):
        """Test tablet-optimized markdown extraction"""
        tablet_config = {
            "url": "https://news-website.com/article",
            "filter_type": "fit",
            "query": "main article content",
            "browser_config": {
                "headless": True,
                "viewport": {
                    "width": 768,
                    "height": 1024,
                    "deviceScaleFactor": 2
                },
                "user_agent": "Mozilla/5.0 (iPad; CPU OS 14_0 like Mac OS X) AppleWebKit/605.1.15"
            },
            "crawler_config": {
                "wait_for": "css:.article-content",
                "excluded_tags": ["nav", "footer", "aside", "ad"],
                "word_count_threshold": 20
            }
        }
        
        req = MarkdownRequest.model_validate(tablet_config)
        
        assert req.browser_config["viewport"]["width"] == 768
        assert "iPad" in req.browser_config["user_agent"]
        assert req.filter_type.value == "fit"


class TestProxyAndPrivacyScenarios:
    """Test proxy, VPN, and privacy-focused scenarios"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_proxy_crawling_configuration(self):
        """Test crawling through proxy with privacy settings"""
        proxy_config = {
            "url": "https://geo-restricted-content.com",
            "filter_type": "bm25",
            "query": "restricted content access",
            "browser_config": {
                "headless": True,
                "proxy": "http://proxy-server.example.com:8080",
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "java_script_enabled": False,  # Reduce fingerprinting
                "ignore_https_errors": True
            },
            "crawler_config": {
                "cache_mode": "bypass",  # Don't cache sensitive content
                "page_timeout": 60000,   # Longer timeout for proxy
                "excluded_tags": ["script", "iframe", "embed"],
                "only_text": True        # Text-only for privacy
            }
        }
        
        req = MarkdownRequest.model_validate(proxy_config)
        
        assert "proxy-server.example.com" in req.browser_config["proxy"]
        assert req.browser_config["java_script_enabled"] is False
        assert req.crawler_config["cache_mode"] == "bypass"
        assert req.crawler_config["only_text"] is True
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_tor_like_configuration(self):
        """Test Tor-like privacy configuration"""
        privacy_config = {
            "url": "https://privacy-sensitive-site.onion",
            "browser_config": {
                "headless": True,
                "proxy": "socks5://localhost:9050",  # Tor proxy
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
                "java_script_enabled": False,
                "light_mode": True,
                "text_mode": True
            },
            "crawler_config": {
                "cache_mode": "disabled",
                "page_timeout": 120000,  # Very long timeout for Tor
                "excluded_tags": ["script", "style", "img", "video", "audio"],
                "only_text": True
            }
        }
        
        req = HTMLRequest.model_validate(privacy_config)
        
        assert "socks5://localhost:9050" in req.browser_config["proxy"]
        assert req.browser_config["text_mode"] is True
        assert req.crawler_config["cache_mode"] == "disabled"


class TestDynamicApplicationScenarios:
    """Test scenarios for modern dynamic web applications"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_spa_interaction_scenario(self):
        """Test Single Page Application interaction"""
        spa_config = {
            "url": "https://modern-spa.app/dashboard",
            "scripts": [
                "// Wait for app initialization",
                "await new Promise(resolve => setTimeout(resolve, 3000))",
                "// Navigate to specific section",
                "document.querySelector('[data-testid=\"reports-tab\"]').click()",
                "await new Promise(resolve => setTimeout(resolve, 2000))",
                "// Extract data after navigation",
                "return document.querySelector('.report-data').textContent"
            ],
            "browser_config": {
                "headless": False,  # Some SPAs detect headless
                "java_script_enabled": True,
                "viewport": {"width": 1366, "height": 768},
                "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            "crawler_config": {
                "session_id": "spa-session-123",
                "wait_for": "css:[data-testid='app-ready']",
                "page_timeout": 90000,  # SPAs can take time to load
                "simulation_id": "dashboard-interaction"
            }
        }
        
        req = JSEndpointRequest.model_validate(spa_config)
        
        assert req.browser_config["headless"] is False
        assert req.browser_config["java_script_enabled"] is True
        assert req.crawler_config["session_id"] == "spa-session-123"
        assert "reports-tab" in req.scripts[2]
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_react_app_state_extraction(self):
        """Test extracting state from React application"""
        react_config = {
            "url": "https://react-dashboard.com",
            "scripts": [
                "// Access React DevTools hook if available",
                "window.React = require('react')",
                "// Wait for components to mount",
                "await new Promise(resolve => setTimeout(resolve, 5000))",
                "// Extract Redux state if available",
                "return window.__REDUX_DEVTOOLS_EXTENSION__ ? window.store.getState() : 'No Redux state found'"
            ],
            "browser_config": {
                "headless": True,
                "java_script_enabled": True,
                "viewport": {"width": 1920, "height": 1080}
            },
            "crawler_config": {
                "wait_for": "css:.react-root",
                "page_timeout": 60000,
                "session_id": "react-extraction"
            }
        }
        
        req = JSEndpointRequest.model_validate(react_config)
        
        assert "React" in req.scripts[0]
        assert "Redux" in req.scripts[3]
        assert req.crawler_config["wait_for"] == "css:.react-root"


class TestDocumentationScenarios:
    """Test scenarios for documentation and content extraction"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_api_documentation_extraction(self):
        """Test extracting API documentation with clean formatting"""
        docs_config = {
            "url": "https://api-docs.service.com/reference",
            "filter_type": "fit",
            "query": "API endpoints methods parameters",
            "browser_config": {
                "headless": True,
                "viewport": {"width": 1200, "height": 800},
                "user_agent": "Mozilla/5.0 Documentation Bot"
            },
            "crawler_config": {
                "wait_for": "css:.api-reference-loaded",
                "excluded_tags": ["nav", "footer", "aside", "ad", "cookie-banner"],
                "word_count_threshold": 15,
                "extraction_strategy": "NoExtractionStrategy",
                "css_selector": ".api-documentation"
            }
        }
        
        req = MarkdownRequest.model_validate(docs_config)
        
        assert req.filter_type.value == "fit"
        assert "API endpoints" in req.query
        assert req.crawler_config["word_count_threshold"] == 15
        assert ".api-documentation" in req.crawler_config["css_selector"]
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_technical_blog_pdf_generation(self):
        """Test generating PDF from technical blog post"""
        blog_pdf_config = {
            "url": "https://tech-blog.com/detailed-tutorial",
            "output_path": "/docs/tutorial.pdf",
            "browser_config": {
                "headless": True,
                "viewport": {"width": 1200, "height": 1600},  # Tall for reading
                "user_agent": "Mozilla/5.0 PDF Generator"
            },
            "crawler_config": {
                "wait_for": "css:.post-content",
                "excluded_tags": ["nav", "footer", "comments", "related-posts"],
                "page_timeout": 45000,
                "pdf": True,
                "css_selector": "article.main-content"
            }
        }
        
        req = PDFRequest.model_validate(blog_pdf_config)
        
        assert req.output_path == "/docs/tutorial.pdf"
        assert req.browser_config["viewport"]["height"] == 1600
        assert "article.main-content" in req.crawler_config["css_selector"]


class TestEcommerceScenarios:
    """Test e-commerce and product-related scenarios"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_product_page_screenshot(self):
        """Test capturing product page screenshots"""
        product_config = {
            "url": "https://shop.example.com/product/123",
            "screenshot_wait_for": 4.0,  # Wait for product images to load
            "output_path": "/tmp/product_screenshot.png",
            "browser_config": {
                "headless": True,
                "viewport": {"width": 1920, "height": 1200},
                "user_agent": "Mozilla/5.0 Product Screenshot Bot"
            },
            "crawler_config": {
                "wait_for": "css:.product-images-loaded",
                "page_timeout": 60000,  # E-commerce sites can be slow
                "excluded_tags": ["chat-widget", "popup", "modal"]
            }
        }
        
        req = ScreenshotRequest.model_validate(product_config)
        
        assert req.screenshot_wait_for == 4.0
        assert req.browser_config["viewport"]["width"] == 1920
        assert req.crawler_config["wait_for"] == "css:.product-images-loaded"
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_price_monitoring_extraction(self):
        """Test extracting pricing information for monitoring"""
        price_config = {
            "url": "https://marketplace.com/item/xyz",
            "filter_type": "bm25",
            "query": "price cost shipping availability stock",
            "browser_config": {
                "headless": True,
                "user_agent": "Mozilla/5.0 (compatible; PriceBot/1.0)",
                "viewport": {"width": 1366, "height": 768}
            },
            "crawler_config": {
                "cache_mode": "bypass",  # Always get fresh prices
                "wait_for": "css:.price-loaded",
                "excluded_tags": ["recommendations", "reviews", "ads"],
                "css_selector": ".product-details, .pricing-info"
            }
        }
        
        req = MarkdownRequest.model_validate(price_config)
        
        assert "price cost shipping" in req.query
        assert req.crawler_config["cache_mode"] == "bypass"
        assert ".pricing-info" in req.crawler_config["css_selector"]


class TestPerformanceTestingScenarios:
    """Test performance and load testing scenarios"""
    
    @pytest.mark.skipif(not SCHEMAS_AVAILABLE, reason="Schemas not available")
    def test_performance_monitoring_configuration(self):
        """Test configuration for performance monitoring"""
        perf_config = {
            "url": "https://target-website.com/heavy-page",
            "scripts": [
                "// Measure page load performance",
                "const perfData = performance.getEntriesByType('navigation')[0]",
                "const loadTime = perfData.loadEventEnd - perfData.fetchStart",
                "const domReady = perfData.domContentLoadedEventEnd - perfData.fetchStart",
                "return { loadTime, domReady, url: window.location.href }"
            ],
            "browser_config": {
                "headless": True,
                "java_script_enabled": True,
                "light_mode": False,  # Full rendering for accurate timing
                "viewport": {"width": 1920, "height": 1080}
            },
            "crawler_config": {
                "page_timeout": 30000,
                "wait_for": "css:body",  # Minimal wait
                "cache_mode": "bypass"   # Fresh load each time
            }
        }
        
        req = JSEndpointRequest.model_validate(perf_config)
        
        assert "performance.getEntriesByType" in req.scripts[1]
        assert req.browser_config["light_mode"] is False
        assert req.crawler_config["cache_mode"] == "bypass"


def run_scenario_tests():
    """Run all real-world scenario tests manually"""
    import unittest
    
    test_classes = [
        TestMobileWebScenarios,
        TestProxyAndPrivacyScenarios,
        TestDynamicApplicationScenarios,
        TestDocumentationScenarios,
        TestEcommerceScenarios,
        TestPerformanceTestingScenarios
    ]
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n🌍 Real-World Scenario Tests Summary:")
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
    success = run_scenario_tests()
    if not success:
        sys.exit(1)
    print("🎉 All real-world scenario tests passed!")