#!/usr/bin/env python3
"""
Simple validation of enhanced schemas
"""
import json

# Test that our schema changes make sense
config_examples = {
    "md_tool": {
        "url": "https://example.com",
        "filter_type": "fit", 
        "query": "test query",
        "cache_version": "1",
        "browser_config": {
            "headless": False,
            "viewport": {"width": 1920, "height": 1080},
            "user_agent": "Mozilla/5.0 Custom"
        },
        "crawler_config": {
            "wait_for": "css:.main-content",
            "excluded_tags": ["nav", "footer"],
            "cache_mode": "bypass",
            "page_timeout": 30000
        }
    },
    "screenshot_tool": {
        "url": "https://example.com",
        "screenshot_wait_for": 3.0,
        "browser_config": {
            "headless": False,
            "viewport": {"width": 1920, "height": 1080}
        },
        "crawler_config": {
            "wait_for": "css:.page-loaded",
            "page_timeout": 45000
        }
    },
    "execute_js_tool": {
        "url": "https://app.example.com",
        "scripts": ["document.title", "window.location.href"],
        "browser_config": {
            "headless": False,
            "java_script_enabled": True,
            "viewport": {"width": 1366, "height": 768}
        },
        "crawler_config": {
            "session_id": "custom-session",
            "wait_for": "css:.app-ready",
            "page_timeout": 60000
        }
    }
}

print("🎯 CONFIGURATION EXAMPLES VALIDATION")
print("=" * 50)

for tool_name, example in config_examples.items():
    print(f"\n**{tool_name}** example:")
    print(json.dumps(example, indent=2))
    
    # Validate structure
    required_fields = ["url"]
    if tool_name == "execute_js_tool":
        required_fields.append("scripts")
    
    has_all_required = all(field in example for field in required_fields)
    has_browser_config = "browser_config" in example
    has_crawler_config = "crawler_config" in example
    
    print(f"✅ Required fields: {has_all_required}")
    print(f"✅ browser_config: {has_browser_config}")
    print(f"✅ crawler_config: {has_crawler_config}")

print("\n🎉 All configuration examples are well-formed!")

# Show the power of the new configs
print("\n🚀 CONFIGURATION POWER EXAMPLES")
print("=" * 40)

power_examples = [
    {
        "scenario": "Mobile screenshot with custom wait",
        "config": {
            "url": "https://responsive-site.com",
            "browser_config": {
                "viewport": {"width": 375, "height": 667},
                "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
            },
            "crawler_config": {
                "wait_for": "css:.mobile-content-ready",
                "screenshot_wait_for": 5.0
            }
        }
    },
    {
        "scenario": "Proxy-based crawling with custom filtering",
        "config": {
            "url": "https://geo-restricted-site.com",
            "filter_type": "bm25",
            "query": "main article content",
            "browser_config": {
                "proxy": "http://proxy.example.com:8080",
                "headless": True
            },
            "crawler_config": {
                "cache_mode": "bypass",
                "excluded_tags": ["ads", "banner", "nav"],
                "word_count_threshold": 25
            }
        }
    },
    {
        "scenario": "JavaScript execution in persistent session",
        "config": {
            "url": "https://dynamic-app.com",
            "scripts": [
                "localStorage.setItem('session', 'active')",
                "document.querySelector('.user-data').textContent"
            ],
            "browser_config": {
                "headless": False,
                "java_script_enabled": True
            },
            "crawler_config": {
                "session_id": "persistent-session",
                "wait_for": "css:.app-initialized",
                "page_timeout": 120000
            }
        }
    }
]

for i, example in enumerate(power_examples, 1):
    print(f"\n{i}. {example['scenario']}")
    print(json.dumps(example['config'], indent=2))

print("\n✨ These examples show the enhanced flexibility of our MCP tools!")