from typing import List, Optional, Dict
from enum import Enum
from pydantic import BaseModel, Field
from utils import FilterType


class CrawlRequest(BaseModel):
    """Request body for the /crawl endpoint."""
    urls: List[str] = Field(
        ...,
        min_length=1, 
        max_length=100,
        description="List of absolute http/https URLs to crawl",
        examples=[
            ["https://example.com"],
            ["https://docs.python.org", "https://github.com/python"],
            ["https://site1.com", "https://site2.com", "https://site3.com"]
        ]
    )
    browser_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Browser configuration options (headless, viewport, etc.)",
        examples=[{}, {"headless": False}, {"viewport": {"width": 1920, "height": 1080}}]
    )
    crawler_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Crawler behavior configuration (cache mode, extraction strategy, etc.)",
        examples=[{}, {"cache_mode": "bypass"}, {"extraction_strategy": "NoExtractionStrategy"}]
    )

class MarkdownRequest(BaseModel):
    """Request body for the /md endpoint."""
    model_config = {"populate_by_name": True}
    url: str = Field(
        ..., 
        description="Absolute http/https URL to fetch",
        examples=["https://example.com", "https://docs.python.org"]
    )
    filter_type: FilterType = Field(
        FilterType.FIT,
        alias="f", 
        description="Content filtering strategy",
        json_schema_extra={
            "enum_descriptions": {
                "raw": "Extract all content without filtering",
                "fit": "Smart content filtering for readability", 
                "bm25": "Use BM25 algorithm with query for targeted extraction",
                "llm": "Use LLM-based filtering with query for intelligent extraction"
            }
        }
    )
    query: Optional[str] = Field(
        None,
        alias="q",
        description="Search query for BM25/LLM filters (required for bm25 and llm filter types)",
        examples=["python tutorial", "installation guide", "API documentation"]
    )
    cache_version: Optional[str] = Field(
        "0",
        alias="c",
        description="Cache version identifier for cache invalidation",
        examples=["0", "1", "v2.1"]
    )
    browser_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Browser configuration options (headless, viewport, proxy, etc.)",
        examples=[
            {},
            {"headless": False},
            {"viewport": {"width": 1920, "height": 1080}},
            {"proxy": "http://proxy:8080", "headless": True},
            {"user_agent": "Mozilla/5.0 Custom Agent", "java_script_enabled": False}
        ]
    )
    crawler_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Crawler behavior configuration (cache mode, extraction strategy, etc.)",
        examples=[
            {},
            {"cache_mode": "bypass"},
            {"extraction_strategy": "NoExtractionStrategy"},
            {"wait_for": "css:.main-content", "page_timeout": 30000},
            {"excluded_tags": ["nav", "footer"], "word_count_threshold": 10}
        ]
    )


class RawCode(BaseModel):
    code: str

class HTMLRequest(BaseModel):
    """Request body for the /html endpoint."""
    url: str = Field(
        ..., 
        description="Absolute http/https URL to fetch and process",
        examples=["https://example.com", "https://docs.python.org", "https://github.com"]
    )
    browser_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Browser configuration options (headless, viewport, proxy, etc.)",
        examples=[
            {},
            {"headless": False},
            {"viewport": {"width": 1920, "height": 1080}},
            {"proxy": "http://proxy:8080", "headless": True}
        ]
    )
    crawler_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Crawler behavior configuration (cache mode, extraction strategy, etc.)",
        examples=[
            {},
            {"cache_mode": "bypass"},
            {"wait_for": "css:.main-content"},
            {"excluded_tags": ["nav", "footer"], "only_text": True}
        ]
    )
    
class ScreenshotRequest(BaseModel):
    """Request body for the /screenshot endpoint."""
    url: str = Field(
        ..., 
        description="Absolute http/https URL to capture as screenshot",
        examples=["https://example.com", "https://github.com/user/repo", "https://docs.python.org"]
    )
    screenshot_wait_for: Optional[float] = Field(
        2.0,
        description="Seconds to wait before capturing screenshot (allows page to load)",
        examples=[1.0, 2.0, 5.0],
        ge=0.1,
        le=30.0
    )
    output_path: Optional[str] = Field(
        None,
        description="Local file path to save screenshot (optional, returns base64 if not provided)",
        examples=["/tmp/screenshot.png", "./output/page.png"]
    )
    browser_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Browser configuration options (headless, viewport, proxy, etc.)",
        examples=[
            {},
            {"headless": False},
            {"viewport": {"width": 1920, "height": 1080}},
            {"user_agent": "Mozilla/5.0 Custom Screenshot Agent"}
        ]
    )
    crawler_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Crawler behavior configuration (wait conditions, timeouts, etc.)",
        examples=[
            {},
            {"wait_for": "css:.main-content"},
            {"page_timeout": 30000},
            {"excluded_tags": ["script", "style"]}
        ]
    )

class PDFRequest(BaseModel):
    """Request body for the /pdf endpoint."""
    url: str = Field(
        ..., 
        description="Absolute http/https URL to convert to PDF",
        examples=["https://example.com", "https://docs.python.org", "https://github.com/user/repo"]
    )
    output_path: Optional[str] = Field(
        None,
        description="Local file path to save PDF (optional, returns base64 if not provided)", 
        examples=["/tmp/document.pdf", "./output/page.pdf"]
    )
    browser_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Browser configuration options (headless, viewport, proxy, etc.)",
        examples=[
            {},
            {"headless": True},
            {"viewport": {"width": 1920, "height": 1080}},
            {"user_agent": "Mozilla/5.0 PDF Generator"}
        ]
    )
    crawler_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Crawler behavior configuration (wait conditions, timeouts, etc.)",
        examples=[
            {},
            {"wait_for": "css:.main-content"},
            {"page_timeout": 30000},
            {"excluded_tags": ["script", "style"], "only_text": False}
        ]
    )


class JSEndpointRequest(BaseModel):
    """Request body for the /execute_js endpoint."""
    url: str = Field(
        ..., 
        description="Absolute http/https URL to execute JavaScript on",
        examples=["https://example.com", "https://app.website.com", "https://dashboard.site.com"]
    )
    scripts: List[str] = Field(
        ...,
        description="List of JavaScript code snippets to execute sequentially on the page",
        examples=[
            ["document.title"],
            ["document.querySelector('h1').textContent", "window.location.href"],
            ["(() => { return Array.from(document.querySelectorAll('a')).map(a => a.href); })()"]
        ],
        min_length=1
    )
    browser_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Browser configuration options (headless, viewport, proxy, etc.)",
        examples=[
            {},
            {"headless": False},
            {"viewport": {"width": 1920, "height": 1080}},
            {"java_script_enabled": True, "user_agent": "Mozilla/5.0 JS Executor"}
        ]
    )
    crawler_config: Optional[Dict] = Field(
        default_factory=dict,
        description="Crawler behavior configuration (wait conditions, timeouts, etc.)",
        examples=[
            {},
            {"wait_for": "css:.dynamic-content"},
            {"page_timeout": 30000},
            {"simulation_id": "custom-session", "session_id": "js-session"}
        ]
    )