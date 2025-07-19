#!/usr/bin/env python3
"""
Tests for MCP resource functionality
Tests config_guide and filter_guide resources
"""
import pytest
import sys
import os
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'deploy', 'docker'))

try:
    from server import get_config_guide, get_filter_guide
    RESOURCES_AVAILABLE = True
except ImportError:
    RESOURCES_AVAILABLE = False
    print("⚠️  MCP resources not available - tests will be skipped")


class TestMCPResourceFunctionality:
    """Test that MCP resources work correctly"""
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_config_guide_structure(self):
        """Test config_guide resource has proper structure"""
        guide = get_config_guide()
        
        # Check top-level structure
        assert isinstance(guide, dict)
        assert "title" in guide
        assert "description" in guide
        assert "browser_config" in guide
        assert "crawler_config" in guide
        assert "tool_specific_examples" in guide
        assert "best_practices" in guide
        
        # Check title and description are strings
        assert isinstance(guide["title"], str)
        assert isinstance(guide["description"], str)
        assert len(guide["title"]) > 0
        assert len(guide["description"]) > 0
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_config_guide_browser_config_section(self):
        """Test browser_config section has all expected options"""
        guide = get_config_guide()
        browser_config = guide["browser_config"]
        
        assert "description" in browser_config
        assert "common_options" in browser_config
        
        common_options = browser_config["common_options"]
        
        # Check essential browser config options are documented
        expected_options = ["headless", "viewport", "user_agent", "proxy", "java_script_enabled"]
        for option in expected_options:
            assert option in common_options, f"Missing browser config option: {option}"
            
            option_info = common_options[option]
            assert "type" in option_info
            assert "description" in option_info
            assert "examples" in option_info
            assert len(option_info["examples"]) > 0
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_config_guide_crawler_config_section(self):
        """Test crawler_config section has all expected options"""
        guide = get_config_guide()
        crawler_config = guide["crawler_config"]
        
        assert "description" in crawler_config
        assert "common_options" in crawler_config
        
        common_options = crawler_config["common_options"]
        
        # Check essential crawler config options are documented
        expected_options = [
            "cache_mode", "wait_for", "page_timeout", "excluded_tags", 
            "extraction_strategy", "word_count_threshold", "only_text"
        ]
        for option in expected_options:
            assert option in common_options, f"Missing crawler config option: {option}"
            
            option_info = common_options[option]
            assert "type" in option_info
            assert "description" in option_info
            assert "examples" in option_info
            assert len(option_info["examples"]) > 0
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_config_guide_tool_examples(self):
        """Test tool-specific examples are complete"""
        guide = get_config_guide()
        tool_examples = guide["tool_specific_examples"]
        
        # Check we have examples for key tools
        expected_tools = ["md", "screenshot", "execute_js"]
        for tool in expected_tools:
            assert tool in tool_examples, f"Missing tool example: {tool}"
            
            tool_info = tool_examples[tool]
            assert "description" in tool_info
            assert "examples" in tool_info
            assert len(tool_info["examples"]) > 0
            
            # Check first example has proper structure
            first_example = tool_info["examples"][0]
            assert "url" in first_example
            assert first_example["url"].startswith("http")
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_config_guide_best_practices(self):
        """Test best practices section is useful"""
        guide = get_config_guide()
        best_practices = guide["best_practices"]
        
        assert isinstance(best_practices, list)
        assert len(best_practices) > 0
        
        # Each practice should be a non-empty string
        for practice in best_practices:
            assert isinstance(practice, str)
            assert len(practice) > 10  # Should be descriptive
            assert practice[0].isupper()  # Should start with capital letter


class TestFilterGuideResource:
    """Test filter_guide resource functionality"""
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_filter_guide_structure(self):
        """Test filter_guide resource has proper structure"""
        guide = get_filter_guide()
        
        # Check top-level structure
        assert isinstance(guide, dict)
        assert "title" in guide
        assert "description" in guide
        assert "filter_types" in guide
        assert "fit_markdown_details" in guide
        assert "choosing_filters" in guide
        assert "advanced_combinations" in guide
        
        # Check title and description
        assert "Content Filtering" in guide["title"]
        assert isinstance(guide["description"], str)
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_filter_types_coverage(self):
        """Test all filter types are documented"""
        guide = get_filter_guide()
        filter_types = guide["filter_types"]
        
        # Check all known filter types are documented
        expected_filters = ["raw", "fit", "bm25", "llm"]
        for filter_type in expected_filters:
            assert filter_type in filter_types, f"Missing filter type: {filter_type}"
            
            filter_info = filter_types[filter_type]
            assert "description" in filter_info
            assert "use_cases" in filter_info
            assert "example" in filter_info
            
            # Check use cases are meaningful
            assert isinstance(filter_info["use_cases"], list)
            assert len(filter_info["use_cases"]) > 0
            
            # Check example has proper structure
            example = filter_info["example"]
            assert "filter_type" in example
            assert "result" in example
            assert example["filter_type"] == filter_type
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_fit_markdown_details(self):
        """Test FIT markdown explanation is comprehensive"""
        guide = get_filter_guide()
        fit_details = guide["fit_markdown_details"]
        
        assert "description" in fit_details
        assert "benefits" in fit_details
        assert "technical_details" in fit_details
        
        # Check description mentions FIT acronym
        assert "Filter, Identify, Transform" in fit_details["description"]
        
        # Check benefits and technical details are lists
        assert isinstance(fit_details["benefits"], list)
        assert isinstance(fit_details["technical_details"], list)
        assert len(fit_details["benefits"]) >= 3
        assert len(fit_details["technical_details"]) >= 3
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_choosing_filters_guidance(self):
        """Test filter selection guidance is helpful"""
        guide = get_filter_guide()
        choosing_filters = guide["choosing_filters"]
        
        # Should have guidance for each filter type
        expected_filters = ["raw", "fit", "bm25", "llm"]
        for filter_type in expected_filters:
            assert filter_type in choosing_filters
            guidance = choosing_filters[filter_type]
            assert isinstance(guidance, str)
            assert len(guidance) > 20  # Should be descriptive
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_advanced_combinations(self):
        """Test advanced combination examples are practical"""
        guide = get_filter_guide()
        combinations = guide["advanced_combinations"]
        
        assert isinstance(combinations, list)
        assert len(combinations) > 0
        
        for combination in combinations:
            assert "scenario" in combination
            assert "recommendation" in combination
            
            scenario = combination["scenario"]
            recommendation = combination["recommendation"]
            
            assert isinstance(scenario, str)
            assert isinstance(recommendation, dict)
            assert "filter_type" in recommendation


class TestResourceSerialization:
    """Test that resources can be properly serialized"""
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_config_guide_json_serializable(self):
        """Test config_guide can be converted to JSON"""
        guide = get_config_guide()
        
        # Should not raise an exception
        json_str = json.dumps(guide, default=str)
        assert len(json_str) > 1000  # Should be substantial
        
        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed["title"] == guide["title"]
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_filter_guide_json_serializable(self):
        """Test filter_guide can be converted to JSON"""
        guide = get_filter_guide()
        
        # Should not raise an exception
        json_str = json.dumps(guide, default=str)
        assert len(json_str) > 1000  # Should be substantial
        
        # Should be deserializable
        parsed = json.loads(json_str)
        assert parsed["title"] == guide["title"]
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_resources_contain_no_functions(self):
        """Test resources contain only serializable data"""
        config_guide = get_config_guide()
        filter_guide = get_filter_guide()
        
        # Recursively check for non-serializable types
        def check_serializable(obj, path=""):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    check_serializable(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_serializable(v, f"{path}[{i}]")
            elif callable(obj):
                raise AssertionError(f"Found function at {path}")
            # Other types (str, int, float, bool, None) are serializable
        
        check_serializable(config_guide, "config_guide")
        check_serializable(filter_guide, "filter_guide")


class TestResourceContent:
    """Test the actual content quality of resources"""
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_config_examples_are_realistic(self):
        """Test that configuration examples are realistic and useful"""
        guide = get_config_guide()
        
        # Check browser config examples
        browser_examples = guide["browser_config"]["common_options"]["viewport"]["examples"]
        
        # Should have common screen resolutions
        viewport_widths = [ex["width"] for ex in browser_examples if isinstance(ex, dict)]
        assert 1920 in viewport_widths  # Common desktop
        assert any(w < 500 for w in viewport_widths)  # Mobile size
        
        # Check user agent examples
        ua_examples = guide["browser_config"]["common_options"]["user_agent"]["examples"]
        assert any("Windows" in ua for ua in ua_examples)
        assert any("iPhone" in ua for ua in ua_examples)
    
    @pytest.mark.skipif(not RESOURCES_AVAILABLE, reason="Resources not available")
    def test_crawler_examples_are_practical(self):
        """Test crawler config examples are practical"""
        guide = get_config_guide()
        
        # Check wait_for examples use real CSS selectors
        wait_examples = guide["crawler_config"]["common_options"]["wait_for"]["examples"]
        for example in wait_examples:
            assert example.startswith("css:")
            assert "." in example or "#" in example  # Should have class or ID selectors
        
        # Check excluded_tags are realistic
        tag_examples = guide["crawler_config"]["common_options"]["excluded_tags"]["examples"]
        all_tags = []
        for example in tag_examples:
            all_tags.extend(example)
        
        # Should include common tags to exclude
        assert "script" in all_tags
        assert "style" in all_tags
        assert "nav" in all_tags


def run_resource_tests():
    """Run all MCP resource tests manually"""
    import unittest
    
    test_classes = [
        TestMCPResourceFunctionality,
        TestFilterGuideResource,
        TestResourceSerialization,
        TestResourceContent
    ]
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n📚 MCP Resource Tests Summary:")
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
    success = run_resource_tests()
    if not success:
        sys.exit(1)
    print("🎉 All MCP resource tests passed!")