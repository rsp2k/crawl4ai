#!/usr/bin/env python3
"""
Tests for MCP prompt functionality
Tests prompt content, structure, and discoverability
"""
import pytest
import sys
import os
import json

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'deploy', 'docker'))

try:
    from server import (
        get_mobile_browser_prompt, get_performance_browser_prompt,
        get_privacy_crawler_prompt, get_ecommerce_crawler_prompt,
        get_spa_configuration_prompt, get_documentation_extraction_prompt
    )
    PROMPTS_AVAILABLE = True
except ImportError:
    PROMPTS_AVAILABLE = False
    print("⚠️  MCP prompts not available - tests will be skipped")


class TestMCPPromptFunctionality:
    """Test that MCP prompts work correctly"""
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_mobile_browser_prompt_structure(self):
        """Test mobile browser prompt has correct structure"""
        prompt = get_mobile_browser_prompt()
        
        # Check top-level structure
        assert isinstance(prompt, dict)
        assert "messages" in prompt
        assert isinstance(prompt["messages"], list)
        assert len(prompt["messages"]) == 2  # user + assistant
        
        # Check user message
        user_msg = prompt["messages"][0]
        assert user_msg["role"] == "user"
        assert "content" in user_msg
        assert "mobile" in user_msg["content"].lower()
        assert "iphone" in user_msg["content"].lower()
        assert "browser_config" in user_msg["content"]
        
        # Check assistant message
        assistant_msg = prompt["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert "content" in assistant_msg
        assert "375" in assistant_msg["content"]  # iPhone width
        assert "deviceScaleFactor" in assistant_msg["content"]
        assert "user_agent" in assistant_msg["content"]
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_performance_browser_prompt_content(self):
        """Test performance monitoring prompt has relevant content"""
        prompt = get_performance_browser_prompt()
        
        assistant_content = prompt["messages"][1]["content"]
        
        # Should include performance-specific configurations
        assert "cache_mode" in assistant_content
        assert "bypass" in assistant_content
        assert "performance.getEntriesByType" in assistant_content
        assert "loadTime" in assistant_content
        assert "domReady" in assistant_content
        
        # Should explain why these settings matter
        assert "Fresh load every time" in assistant_content
        assert "Full rendering" in assistant_content
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_privacy_crawler_prompt_security_features(self):
        """Test privacy prompt includes security features"""
        prompt = get_privacy_crawler_prompt()
        
        assistant_content = prompt["messages"][1]["content"]
        
        # Should include privacy-specific features
        assert "socks5://" in assistant_content
        assert "java_script_enabled" in assistant_content
        assert "false" in assistant_content  # JS disabled
        assert "cache_mode" in assistant_content
        assert "disabled" in assistant_content
        assert "only_text" in assistant_content
        
        # Should explain privacy considerations
        assert "Tor" in assistant_content
        assert "fingerprinting" in assistant_content
        assert "tracking" in assistant_content
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_ecommerce_crawler_prompt_product_focus(self):
        """Test e-commerce prompt focuses on product extraction"""
        prompt = get_ecommerce_crawler_prompt()
        
        user_content = prompt["messages"][0]["content"]
        assistant_content = prompt["messages"][1]["content"]
        
        # User should ask about product-specific needs
        assert "product" in user_content.lower()
        assert "price" in user_content.lower()
        assert "availability" in user_content.lower()
        
        # Assistant should provide product-optimized config
        assert "bm25" in assistant_content
        assert "price cost shipping" in assistant_content
        assert ".product-" in assistant_content
        assert ".pricing-" in assistant_content
        assert "document.querySelector('.price')" in assistant_content
        
        # Should explain e-commerce considerations
        assert "Dynamic pricing" in assistant_content
        assert "Fresh prices" in assistant_content
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_spa_configuration_prompt_complexity(self):
        """Test SPA prompt handles complex application scenarios"""
        prompt = get_spa_configuration_prompt()
        
        user_content = prompt["messages"][0]["content"]
        assistant_content = prompt["messages"][1]["content"]
        
        # User should describe SPA challenges
        assert "Single Page Application" in user_content
        assert "client-side routing" in user_content.lower()
        assert "dynamic" in user_content.lower()
        
        # Assistant should provide SPA-specific solutions
        assert "headless" in assistant_content
        assert "false" in assistant_content  # Non-headless for SPAs
        assert "session_id" in assistant_content
        assert "data-testid" in assistant_content
        assert "setTimeout" in assistant_content
        assert "click()" in assistant_content
        
        # Should explain SPA considerations
        assert "SPAs take time" in assistant_content
        assert "client-side navigation" in assistant_content
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_documentation_extraction_prompt_clean_output(self):
        """Test documentation prompt optimizes for clean extraction"""
        prompt = get_documentation_extraction_prompt()
        
        user_content = prompt["messages"][0]["content"]
        assistant_content = prompt["messages"][1]["content"]
        
        # User should ask for documentation-specific needs
        assert "documentation" in user_content.lower()
        assert "code examples" in user_content.lower()
        assert "clean" in user_content.lower()
        
        # Assistant should provide doc-optimized config
        assert "filter_type" in assistant_content
        assert "fit" in assistant_content
        assert "excluded_tags" in assistant_content
        assert '["nav", "footer"' in assistant_content
        assert "api-documentation" in assistant_content
        assert "preserve_code" in assistant_content
        
        # Should explain documentation considerations
        assert "FIT filter" in assistant_content
        assert "syntax highlighting" in assistant_content


class TestPromptMessageValidation:
    """Test that prompt messages follow correct format"""
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_all_prompts_have_valid_message_structure(self):
        """Test all prompts follow MCP message structure"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            prompt = prompt_fn()
            
            # Each prompt should have messages array
            assert "messages" in prompt
            assert isinstance(prompt["messages"], list)
            assert len(prompt["messages"]) >= 1
            
            # Each message should have role and content
            for message in prompt["messages"]:
                assert "role" in message
                assert "content" in message
                assert message["role"] in ["user", "assistant", "system"]
                assert isinstance(message["content"], str)
                assert len(message["content"]) > 10  # Should be substantial
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompts_are_json_serializable(self):
        """Test all prompts can be serialized to JSON"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            prompt = prompt_fn()
            
            # Should serialize without error
            json_str = json.dumps(prompt)
            assert len(json_str) > 100  # Should be substantial
            
            # Should deserialize correctly
            parsed = json.loads(json_str)
            assert parsed["messages"] == prompt["messages"]
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompts_contain_realistic_configurations(self):
        """Test prompts contain valid JSON configurations"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            prompt = prompt_fn()
            assistant_content = None
            
            # Find assistant message
            for message in prompt["messages"]:
                if message["role"] == "assistant":
                    assistant_content = message["content"]
                    break
            
            assert assistant_content is not None
            
            # Should contain JSON code blocks
            assert "```json" in assistant_content
            
            # Should contain realistic configuration keys
            assert "browser_config" in assistant_content or "crawler_config" in assistant_content
            assert "url" in assistant_content
            
            # Should contain explanations
            assert ":" in assistant_content  # Explanations with colons
            assert "-" in assistant_content   # Bullet points


class TestPromptContentQuality:
    """Test the quality and usefulness of prompt content"""
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompts_provide_complete_examples(self):
        """Test prompts provide complete, usable examples"""
        prompt_functions = [
            ("mobile", get_mobile_browser_prompt),
            ("performance", get_performance_browser_prompt),
            ("privacy", get_privacy_crawler_prompt),
            ("ecommerce", get_ecommerce_crawler_prompt),
            ("spa", get_spa_configuration_prompt),
            ("documentation", get_documentation_extraction_prompt)
        ]
        
        for name, prompt_fn in prompt_functions:
            prompt = prompt_fn()
            assistant_msg = next(m for m in prompt["messages"] if m["role"] == "assistant")
            content = assistant_msg["content"]
            
            # Should have complete URL example
            assert "https://" in content, f"{name} prompt missing URL example"
            
            # Should explain configuration choices
            assert "**" in content, f"{name} prompt missing bold explanations"
            
            # Should have practical considerations
            explanation_indicators = ["Key", "considerations", "features", "explained"]
            has_explanations = any(indicator in content for indicator in explanation_indicators)
            assert has_explanations, f"{name} prompt missing explanations"
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompts_address_real_world_challenges(self):
        """Test prompts address actual challenges users face"""
        # Mobile prompt should address device simulation
        mobile = get_mobile_browser_prompt()
        mobile_content = mobile["messages"][1]["content"]
        assert "retina" in mobile_content.lower()
        assert "touch events" in mobile_content.lower()
        
        # Performance prompt should address measurement accuracy
        perf = get_performance_browser_prompt()
        perf_content = perf["messages"][1]["content"]
        assert "fresh load" in perf_content.lower()
        assert "timing" in perf_content.lower()
        
        # Privacy prompt should address anonymity
        privacy = get_privacy_crawler_prompt()
        privacy_content = privacy["messages"][1]["content"]
        assert "anonymous" in privacy_content.lower() or "fingerprint" in privacy_content.lower()
        
        # E-commerce prompt should address bot detection
        ecom = get_ecommerce_crawler_prompt()
        ecom_content = ecom["messages"][1]["content"]
        assert "anti-bot" in ecom_content.lower() or "dynamic" in ecom_content.lower()
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompts_include_best_practices(self):
        """Test prompts include configuration best practices"""
        prompt_functions = [
            get_mobile_browser_prompt,
            get_performance_browser_prompt,
            get_privacy_crawler_prompt,
            get_ecommerce_crawler_prompt,
            get_spa_configuration_prompt,
            get_documentation_extraction_prompt
        ]
        
        for prompt_fn in prompt_functions:
            prompt = prompt_fn()
            assistant_content = prompt["messages"][1]["content"]
            
            # Should include timeout considerations
            has_timeout = any(word in assistant_content.lower() 
                            for word in ["timeout", "slow", "wait"])
            assert has_timeout, f"Prompt {prompt_fn.__name__} missing timeout guidance"
            
            # Should include viewport considerations
            has_viewport = "viewport" in assistant_content
            assert has_viewport, f"Prompt {prompt_fn.__name__} missing viewport guidance"


class TestPromptIntegration:
    """Test prompt integration with MCP system"""
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompt_functions_have_docstrings(self):
        """Test all prompt functions have proper docstrings"""
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
            assert len(docstring.strip()) > 20, f"{prompt_fn.__name__} docstring too short"
            assert "Prompt template" in docstring, f"{prompt_fn.__name__} docstring missing template description"
    
    @pytest.mark.skipif(not PROMPTS_AVAILABLE, reason="Prompts not available")
    def test_prompts_complement_tools_and_resources(self):
        """Test prompts complement existing tools and resources"""
        # Prompts should reference tools that exist
        ecom_prompt = get_ecommerce_crawler_prompt()
        ecom_content = ecom_prompt["messages"][1]["content"]
        assert "execute_js" in ecom_content  # References existing tool
        
        doc_prompt = get_documentation_extraction_prompt()
        doc_content = doc_prompt["messages"][1]["content"]
        assert "filter_type" in doc_content  # References tool parameters
        
        # Prompts should use configurations that resources document
        mobile_prompt = get_mobile_browser_prompt()
        mobile_content = mobile_prompt["messages"][1]["content"]
        assert "browser_config" in mobile_content
        assert "crawler_config" in mobile_content


def run_prompt_tests():
    """Run all MCP prompt tests manually"""
    import unittest
    
    test_classes = [
        TestMCPPromptFunctionality,
        TestPromptMessageValidation,
        TestPromptContentQuality,
        TestPromptIntegration
    ]
    
    test_loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = test_loader.loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    print(f"\n💬 MCP Prompt Tests Summary:")
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
    success = run_prompt_tests()
    if not success:
        sys.exit(1)
    print("🎉 All MCP prompt tests passed!")