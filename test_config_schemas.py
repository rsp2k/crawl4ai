#!/usr/bin/env python3
"""
Test the enhanced MCP schemas with configuration support
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'deploy', 'docker'))

from schemas import MarkdownRequest, HTMLRequest, ScreenshotRequest, PDFRequest, JSEndpointRequest

print("🎯 TESTING ENHANCED CONFIG SCHEMAS")
print("=" * 50)

# Test MarkdownRequest with config parameters
print("1. **md** Tool Schema with configs")
try:
    md_req = MarkdownRequest.model_validate({
        "url": "https://example.com",
        "filter_type": "fit",
        "browser_config": {"headless": False, "viewport": {"width": 1920, "height": 1080}},
        "crawler_config": {"wait_for": "css:.main-content", "excluded_tags": ["nav", "footer"]}
    })
    print(f"   ✅ SUCCESS: Enhanced configs accepted")
    
    # Test backwards compatibility
    md_req_old = MarkdownRequest.model_validate({
        "url": "https://example.com", 
        "f": "fit",
        "q": "test query",
        "c": "1"
    })
    print(f"   ✅ SUCCESS: Backwards compatibility maintained")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")

print()

# Test other tools
tools = [
    ("html", HTMLRequest, {
        "url": "https://example.com",
        "browser_config": {"headless": True},
        "crawler_config": {"wait_for": "css:.content"}
    }),
    ("screenshot", ScreenshotRequest, {
        "url": "https://example.com",
        "screenshot_wait_for": 2.0,
        "browser_config": {"viewport": {"width": 1920, "height": 1080}},
        "crawler_config": {"wait_for": "css:.loaded"}
    }),
    ("pdf", PDFRequest, {
        "url": "https://example.com",
        "browser_config": {"headless": True},
        "crawler_config": {"excluded_tags": ["script", "style"]}
    }),
    ("execute_js", JSEndpointRequest, {
        "url": "https://example.com",
        "scripts": ["document.title"],
        "browser_config": {"java_script_enabled": True},
        "crawler_config": {"session_id": "test-session"}
    })
]

for i, (name, schema_class, test_data) in enumerate(tools, 2):
    print(f"{i}. **{name}** Tool Schema with configs")
    try:
        req = schema_class.model_validate(test_data)
        print(f"   ✅ SUCCESS: Enhanced configs accepted")
        
        # Check schema has config fields
        schema = schema_class.model_json_schema()
        props = schema.get('properties', {})
        has_browser_config = 'browser_config' in props
        has_crawler_config = 'crawler_config' in props
        
        print(f"   📋 browser_config: {'✅' if has_browser_config else '❌'}")
        print(f"   📋 crawler_config: {'✅' if has_crawler_config else '❌'}")
        
    except Exception as e:
        print(f"   ❌ FAILED: {e}")
    print()

print("🎉 Enhanced config schema tests completed!")

# Test resource schema generation
print("\n🔧 TESTING MCP RESOURCE FUNCTIONS")
print("=" * 30)

try:
    sys.path.append(os.path.join(os.path.dirname(__file__), 'deploy', 'docker'))
    from server import get_config_guide, get_filter_guide
    
    config_guide = get_config_guide()
    filter_guide = get_filter_guide()
    
    print(f"✅ config_guide: {len(config_guide)} sections")
    print(f"✅ filter_guide: {len(filter_guide)} sections")
    print("✅ MCP resources generate successfully")
    
except Exception as e:
    print(f"❌ MCP resource test failed: {e}")

print("\n✅ All enhanced configuration tests completed!")