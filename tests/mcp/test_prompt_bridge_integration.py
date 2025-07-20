#!/usr/bin/env python3
"""
Tests for MCP prompt bridge integration
Tests that prompts work correctly with the MCP bridge system
"""
import pytest
import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'deploy', 'docker'))

try:
    from mcp_bridge import mcp_prompt
    from server import (
        get_mobile_browser_prompt, get_performance_browser_prompt,
        get_privacy_crawler_prompt, get_ecommerce_crawler_prompt,
        get_spa_configuration_prompt, get_documentation_extraction_prompt
    )
    BRIDGE_AVAILABLE = True
except ImportError:
    BRIDGE_AVAILABLE = False
    print("⚠️  MCP bridge or prompts not available - tests will be skipped")


class TestMCPPromptDecoratorFunctionality:
    """Test the @mcp_prompt decorator functionality"""
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_mcp_prompt_decorator_adds_metadata(self):
        """Test @mcp_prompt decorator adds proper metadata"""
        
        @mcp_prompt("test_prompt")
        def test_prompt_function():
            """Test prompt for validation"""
            return {"messages": [{"role": "user", "content": "test"}]}
        
        # Should add MCP metadata
        assert hasattr(test_prompt_function, '__mcp_kind__')
        assert hasattr(test_prompt_function, '__mcp_name__')
        assert test_prompt_function.__mcp_kind__ == "prompt"
        assert test_prompt_function.__mcp_name__ == "test_prompt"
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_mcp_prompt_decorator_without_name(self):
        """Test @mcp_prompt decorator works without explicit name"""
        
        @mcp_prompt()
        def another_test_prompt():
            """Another test prompt"""
            return {"messages": [{"role": "user", "content": "test"}]}
        
        # Should add metadata with None name (bridge will infer)
        assert hasattr(another_test_prompt, '__mcp_kind__')
        assert hasattr(another_test_prompt, '__mcp_name__')
        assert another_test_prompt.__mcp_kind__ == "prompt"
        assert another_test_prompt.__mcp_name__ is None
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_existing_prompts_have_decorator_metadata(self):
        """Test all existing prompts have proper decorator metadata"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            assert hasattr(prompt_fn, '__mcp_kind__'), f"{prompt_fn.__name__} missing __mcp_kind__"
            assert hasattr(prompt_fn, '__mcp_name__'), f"{prompt_fn.__name__} missing __mcp_name__"
            assert prompt_fn.__mcp_kind__ == "prompt", f"{prompt_fn.__name__} wrong kind"
            assert prompt_fn.__mcp_name__ is not None, f"{prompt_fn.__name__} missing name"


class TestPromptBridgeRegistration:
    """Test prompt registration in MCP bridge"""
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_prompt_registration_logic(self):
        """Test the prompt registration logic in attach_mcp"""
        # This would be tested in integration tests with actual FastAPI app
        # For now, we test the decorator and function structure
        
        # Test that prompts return proper structure for MCP bridge
        mobile_prompt = get_mobile_browser_prompt()
        
        # Should return dict with messages that MCP bridge can process
        assert isinstance(mobile_prompt, dict)
        assert "messages" in mobile_prompt
        assert isinstance(mobile_prompt["messages"], list)
        
        for message in mobile_prompt["messages"]:
            assert "role" in message
            assert "content" in message
            assert message["role"] in ["user", "assistant", "system"]


class TestPromptMCPCompliance:
    """Test prompts comply with MCP specification"""
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_prompt_names_are_valid_identifiers(self):
        """Test prompt names are valid MCP identifiers"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            name = prompt_fn.__mcp_name__
            assert name is not None
            assert isinstance(name, str)
            assert len(name) > 0
            
            # Should be valid identifier (no spaces, special chars except underscore)
            assert name.replace('_', '').replace('-', '').isalnum(), f"Invalid prompt name: {name}"
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_prompt_output_format_mcp_compatible(self):
        """Test prompt output format is MCP-compatible"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            result = prompt_fn()
            
            # Should return structure compatible with MCP GetPromptResult
            assert isinstance(result, dict)
            assert "messages" in result
            assert isinstance(result["messages"], list)
            
            for message in result["messages"]:
                # Each message should have role and content
                assert "role" in message
                assert "content" in message
                
                # Role should be valid MCP role
                assert message["role"] in ["user", "assistant", "system"]
                
                # Content should be string (for MCP TextContent)
                assert isinstance(message["content"], str)
                assert len(message["content"]) > 0


class TestPromptDiscoverability:
    """Test that prompts are discoverable through MCP"""
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_prompt_docstrings_provide_descriptions(self):
        """Test prompt docstrings provide useful descriptions for MCP discovery"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            docstring = prompt_fn.__doc__
            assert docstring is not None, f"{prompt_fn.__name__} missing docstring"
            
            # Docstring should be descriptive enough for MCP list_prompts
            assert len(docstring.strip()) > 30, f"{prompt_fn.__name__} docstring too short"
            
            # Should describe what the prompt helps with
            helpful_words = ["configure", "configuration", "help", "template", "guide"]
            has_helpful_description = any(word in docstring.lower() for word in helpful_words)
            assert has_helpful_description, f"{prompt_fn.__name__} docstring not helpful"
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_prompt_names_are_descriptive(self):
        """Test prompt names clearly indicate their purpose"""
        expected_prompts = {
            "browser_config_mobile": get_mobile_browser_prompt,
            "browser_config_performance": get_performance_browser_prompt,
            "crawler_config_privacy": get_privacy_crawler_prompt,
            "crawler_config_ecommerce": get_ecommerce_crawler_prompt,
            "advanced_spa_config": get_spa_configuration_prompt,
            "documentation_extraction": get_documentation_extraction_prompt
        }
        
        for expected_name, prompt_fn in expected_prompts.items():
            actual_name = prompt_fn.__mcp_name__
            assert actual_name == expected_name, f"Expected {expected_name}, got {actual_name}"
            
            # Name should indicate the domain/purpose
            if "browser_config" in expected_name:
                assert "browser" in expected_name.lower()
            if "crawler_config" in expected_name:
                assert "crawler" in expected_name.lower()
            if "mobile" in expected_name:
                assert "mobile" in expected_name.lower()


class TestPromptErrorHandling:
    """Test prompt error handling and edge cases"""
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_prompts_handle_no_arguments(self):
        """Test prompts work when called without arguments"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            # Should not raise exception when called without args
            try:
                result = prompt_fn()
                assert result is not None
                assert "messages" in result
            except Exception as e:
                pytest.fail(f"{prompt_fn.__name__} raised exception: {e}")
    
    @pytest.mark.skipif(not BRIDGE_AVAILABLE, reason="Bridge not available")
    def test_prompts_return_consistent_structure(self):
        """Test all prompts return consistent structure"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        expected_structure = None
        
        for prompt_fn in prompt_functions:
            result = prompt_fn()
            
            # All prompts should have same top-level structure
            if expected_structure is None:
                expected_structure = set(result.keys())
            else:
                assert set(result.keys()) == expected_structure, f"{prompt_fn.__name__} has different structure"
            
            # All should have messages with same message structure
            for message in result["messages"]:
                assert set(message.keys()) == {"role", "content"}


def run_bridge_integration_tests():
    """Run all MCP prompt bridge integration tests manually"""
    import unittest
    
    test_classes = [
        TestMCPPromptDecoratorFunctionality,
        TestPromptBridgeRegistration,
        TestPromptMCPCompliance,
        TestPromptDiscoverability,
        TestPromptErrorHandling
    ]
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n🔗 MCP Prompt Bridge Integration Tests Summary:")
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
    success = run_bridge_integration_tests()
    if not success:
        sys.exit(1)
    print("🎉 All MCP prompt bridge integration tests passed!")