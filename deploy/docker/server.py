# ───────────────────────── server.py ─────────────────────────
"""
Crawl4AI FastAPI entry‑point
• Browser pool + global page cap
• Rate‑limiting, security, metrics
• /crawl, /crawl/stream, /md, /llm endpoints
"""

# ── stdlib & 3rd‑party imports ───────────────────────────────
from crawler_pool import get_crawler, close_all, janitor
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from auth import create_access_token, get_token_dependency, TokenRequest
from pydantic import BaseModel
from typing import Optional, List, Dict
from fastapi import Request, Depends
from fastapi.responses import FileResponse
import base64
import re
from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
from api import (
    handle_markdown_request, handle_llm_qa,
    handle_stream_crawl_request, handle_crawl_request,
    stream_results
)
from schemas import (
    CrawlRequest,
    MarkdownRequest,
    RawCode,
    HTMLRequest,
    ScreenshotRequest,
    PDFRequest,
    JSEndpointRequest,
)

from utils import (
    FilterType, load_config, setup_logging, verify_email_domain
)
import os
import sys
import time
import asyncio
from typing import List
from contextlib import asynccontextmanager
import pathlib

from fastapi import (
    FastAPI, HTTPException, Request, Path, Query, Depends
)
from rank_bm25 import BM25Okapi
from fastapi.responses import (
    StreamingResponse, RedirectResponse, PlainTextResponse, JSONResponse
)
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.staticfiles import StaticFiles
from job import init_job_router

from mcp_bridge import attach_mcp, mcp_resource, mcp_template, mcp_tool, mcp_prompt

import ast
import crawl4ai as _c4
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address
from prometheus_fastapi_instrumentator import Instrumentator
from redis import asyncio as aioredis

# ── internal imports (after sys.path append) ─────────────────
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

# ────────────────── configuration / logging ──────────────────
config = load_config()
setup_logging(config)

__version__ = "0.5.1-d1"

# ── global page semaphore (hard cap) ─────────────────────────
MAX_PAGES = config["crawler"]["pool"].get("max_pages", 30)
GLOBAL_SEM = asyncio.Semaphore(MAX_PAGES)

# import logging
# page_log = logging.getLogger("page_cap")
# orig_arun = AsyncWebCrawler.arun
# async def capped_arun(self, *a, **kw):
#     await GLOBAL_SEM.acquire()                        # ← take slot
#     try:
#         in_flight = MAX_PAGES - GLOBAL_SEM._value     # used permits
#         page_log.info("🕸️  pages_in_flight=%s / %s", in_flight, MAX_PAGES)
#         return await orig_arun(self, *a, **kw)
#     finally:
#         GLOBAL_SEM.release()                          # ← free slot

orig_arun = AsyncWebCrawler.arun


async def capped_arun(self, *a, **kw):
    async with GLOBAL_SEM:
        return await orig_arun(self, *a, **kw)
AsyncWebCrawler.arun = capped_arun

# ───────────────────── FastAPI lifespan ──────────────────────


@asynccontextmanager
async def lifespan(_: FastAPI):
    await get_crawler(BrowserConfig(
        extra_args=config["crawler"]["browser"].get("extra_args", []),
        **config["crawler"]["browser"].get("kwargs", {}),
    ))           # warm‑up
    app.state.janitor = asyncio.create_task(janitor())        # idle GC
    yield
    app.state.janitor.cancel()
    await close_all()

# ───────────────────── FastAPI instance ──────────────────────
app = FastAPI(
    title=config["app"]["title"],
    version=config["app"]["version"],
    lifespan=lifespan,
)

# ── static playground ──────────────────────────────────────
STATIC_DIR = pathlib.Path(__file__).parent / "static" / "playground"
if not STATIC_DIR.exists():
    raise RuntimeError(f"Playground assets not found at {STATIC_DIR}")
app.mount(
    "/playground",
    StaticFiles(directory=STATIC_DIR, html=True),
    name="play",
)


@app.get("/")
async def root():
    return RedirectResponse("/playground")

# ─────────────────── infra / middleware  ─────────────────────
redis = aioredis.from_url(config["redis"].get("uri", "redis://localhost"))

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[config["rate_limiting"]["default_limit"]],
    storage_uri=config["rate_limiting"]["storage_uri"],
)


def _setup_security(app_: FastAPI):
    sec = config["security"]
    if not sec["enabled"]:
        return
    if sec.get("https_redirect"):
        app_.add_middleware(HTTPSRedirectMiddleware)
    if sec.get("trusted_hosts", []) != ["*"]:
        app_.add_middleware(
            TrustedHostMiddleware, allowed_hosts=sec["trusted_hosts"]
        )


_setup_security(app)

if config["observability"]["prometheus"]["enabled"]:
    Instrumentator().instrument(app).expose(app)

token_dep = get_token_dependency(config)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    resp = await call_next(request)
    if config["security"]["enabled"]:
        resp.headers.update(config["security"]["headers"])
    return resp

# ───────────────── safe config‑dump helper ─────────────────
ALLOWED_TYPES = {
    "CrawlerRunConfig": CrawlerRunConfig,
    "BrowserConfig": BrowserConfig,
}


def _safe_eval_config(expr: str) -> dict:
    """
    Accept exactly one top‑level call to CrawlerRunConfig(...) or BrowserConfig(...).
    Whatever is inside the parentheses is fine *except* further function calls
    (so no  __import__('os') stuff).  All public names from crawl4ai are available
    when we eval.
    """
    tree = ast.parse(expr, mode="eval")

    # must be a single call
    if not isinstance(tree.body, ast.Call):
        raise ValueError("Expression must be a single constructor call")

    call = tree.body
    if not (isinstance(call.func, ast.Name) and call.func.id in {"CrawlerRunConfig", "BrowserConfig"}):
        raise ValueError(
            "Only CrawlerRunConfig(...) or BrowserConfig(...) are allowed")

    # forbid nested calls to keep the surface tiny
    for node in ast.walk(call):
        if isinstance(node, ast.Call) and node is not call:
            raise ValueError("Nested function calls are not permitted")

    # expose everything that crawl4ai exports, nothing else
    safe_env = {name: getattr(_c4, name)
                for name in dir(_c4) if not name.startswith("_")}
    obj = eval(compile(tree, "<config>", "eval"),
               {"__builtins__": {}}, safe_env)
    return obj.dump()


# ── job router ──────────────────────────────────────────────
app.include_router(init_job_router(redis, config, token_dep))

# ──────────────────────── Endpoints ──────────────────────────
@app.post("/token")
async def get_token(req: TokenRequest):
    if not verify_email_domain(req.email):
        raise HTTPException(400, "Invalid email domain")
    token = create_access_token({"sub": req.email})
    return {"email": req.email, "access_token": token, "token_type": "bearer"}


@app.post("/config/dump")
async def config_dump(raw: RawCode):
    try:
        return JSONResponse(_safe_eval_config(raw.code.strip()))
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/md")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool(
    "md",
    title="Markdown Extractor",
    readOnlyHint=True,
    openWorldHint=True,
    idempotentHint=True
)
async def get_markdown(
    request: Request,
    body: MarkdownRequest,
    _td: Dict = Depends(token_dep),
):
    """
    🔄 **Web-to-Markdown Converter & Content Extractor**
    
    Converts any webpage into clean, readable markdown format with intelligent content filtering.
    Perfect for extracting articles, documentation, blog posts, and structured content for AI processing.
    
    **📋 CORE FUNCTIONALITY:**
    • Fetches webpage content and converts HTML to clean markdown
    • Applies intelligent filtering to remove ads, navigation, and boilerplate
    • Extracts main content while preserving formatting and structure
    • Returns structured JSON with original URL, filter settings, and markdown content
    
    **🎛️ FILTER STRATEGIES (filter_type):**
    
    **"fit"** (RECOMMENDED) - Smart content extraction using AI algorithms
    ├─ Removes ads, navigation, sidebars, and promotional content
    ├─ Identifies and preserves main article/content areas 
    ├─ Best for: Blog posts, news articles, documentation, clean reading
    └─ Query parameter: Not required (but can enhance results)
    
    **"raw"** - Complete page content without filtering
    ├─ Converts entire HTML to markdown with no content removal
    ├─ Preserves all page elements including navigation and ads
    ├─ Best for: Complete page archival, custom filtering later
    └─ Query parameter: Ignored
    
    **"bm25"** - Keyword-based content filtering using BM25 algorithm
    ├─ Extracts content sections most relevant to your search query
    ├─ Uses advanced search ranking to find topically relevant content
    ├─ Best for: Research, extracting specific information topics
    └─ Query parameter: REQUIRED - provide keywords you're looking for
    
    **"llm"** - AI-powered intelligent content selection
    ├─ Uses large language models to understand content relevance  
    ├─ Applies contextual understanding for complex filtering needs
    ├─ Best for: Complex content analysis, context-aware extraction
    └─ Query parameter: REQUIRED - describe what content you want
    
    **⚙️ CONFIGURATION OPTIONS:**
    
    **browser_config** - Controls browser behavior:
    • headless: true/false (show browser window)
    • viewport: {width: 1920, height: 1080} (screen size)
    • user_agent: "custom string" (browser identification)
    • proxy: "http://proxy:8080" (proxy server)
    
    **crawler_config** - Controls page processing:
    • wait_for: "css:.content-loaded" (wait for elements)
    • page_timeout: 60000 (max wait time in ms)
    • excluded_tags: ["nav", "footer"] (HTML tags to remove)
    • cache_mode: "enabled/disabled/bypass" (caching strategy)
    
    **💡 COMMON USE CASES:**
    
    📰 **News/Blog Articles**: Use filter_type="fit" for clean, readable content
    📚 **Documentation**: Use filter_type="fit" with excluded_tags=["nav","sidebar"] 
    🔍 **Research**: Use filter_type="bm25" with query="specific topic keywords"
    🤖 **AI Analysis**: Use filter_type="llm" with query="extract technical specifications"
    📊 **Data Collection**: Use filter_type="raw" for complete page capture
    
    **✅ SUCCESS RESPONSE:**
    Returns JSON with: url, filter, query, cache, markdown (content), success=true
    
    **❌ ERROR HANDLING:**
    • Invalid URLs return 400 error
    • Network failures return 500 with error details
    • Timeout issues automatically handled with retries
    """
    try:
        if not body.url.startswith(("http://", "https://")):
            raise HTTPException(
                400, "URL must be absolute and start with http/https")
        markdown = await handle_markdown_request(
            body.url, body.filter_type, body.query, body.cache_version, config,
            body.browser_config, body.crawler_config
        )
        return JSONResponse({
            "url": body.url,
            "filter": body.filter_type,
            "query": body.query,
            "cache": body.cache_version,
            "markdown": markdown,
            "success": True
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to extract markdown: {str(e)}")


@app.post("/html")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool(
    "html",
    title="HTML Content Extractor", 
    readOnlyHint=True,
    openWorldHint=True,
    idempotentHint=True
)
async def generate_html(
    request: Request,
    body: HTMLRequest,
    _td: Dict = Depends(token_dep),
):
    """
    🌐 **HTML Structure Extractor & Schema Builder**
    
    Extracts and preprocesses raw HTML content from webpages for structured data analysis.
    Perfect for AI schema generation, data extraction pipelines, and HTML structure analysis.
    
    **📋 CORE FUNCTIONALITY:**
    • Fetches complete webpage HTML content 
    • Preprocesses HTML for optimal schema extraction
    • Sanitizes and normalizes HTML structure
    • Removes problematic elements that interfere with parsing
    • Returns clean, structured HTML ready for AI processing
    
    **🎯 PRIMARY USE CASES:**
    
    🤖 **AI Schema Generation**: Extract HTML structure for LLMs to build data schemas
    🏗️ **Data Pipeline Input**: Clean HTML for structured data extraction workflows  
    📊 **Content Structure Analysis**: Understand page layout and element hierarchy
    🔍 **Element Discovery**: Find specific HTML patterns and structures
    📝 **Template Analysis**: Study page templates and recurring patterns
    
    **⚙️ CONFIGURATION OPTIONS:**
    
    **browser_config** - Controls browser behavior:
    • headless: true/false (show browser for debugging)
    • viewport: {width: 1920, height: 1080} (affects responsive layouts)
    • user_agent: "custom agent" (for specific content access)
    • java_script_enabled: true/false (run JS for dynamic content)
    
    **crawler_config** - Controls HTML processing:
    • wait_for: "css:.content-loaded" (ensure dynamic content loads)
    • page_timeout: 60000 (max load time in milliseconds)
    • excluded_tags: ["script", "style"] (remove unwanted elements)
    • cache_mode: "enabled/disabled/bypass" (caching strategy)
    
    **💡 WHEN TO USE HTML vs MD TOOL:**
    
    ✅ **Use HTML tool when:**
    • Building data extraction schemas
    • Analyzing page structure and layout
    • Need complete HTML with all elements
    • Creating scrapers or parsers
    • Studying responsive design implementations
    
    ❌ **Use MD tool instead when:**
    • You want readable content for humans/AI
    • Need clean text without HTML markup
    • Extracting articles or documentation
    • Content analysis rather than structure analysis
    
    **✅ SUCCESS RESPONSE:**
    Returns JSON with: html (preprocessed content), url, success=true
    
    **🔧 PREPROCESSING INCLUDES:**
    • Normalized whitespace and formatting
    • Removed problematic script and style elements  
    • Standardized attribute formatting
    • Optimized for machine parsing and analysis
    """
    # Build browser config with defaults and user overrides
    browser_cfg = BrowserConfig()
    if body.browser_config:
        browser_cfg = BrowserConfig.from_kwargs(body.browser_config)

    # Build crawler config with defaults and user overrides
    crawler_cfg = CrawlerRunConfig()
    if body.crawler_config:
        crawler_cfg = CrawlerRunConfig.from_kwargs(body.crawler_config)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        results = await crawler.arun(url=body.url, config=crawler_cfg)
    raw_html = results[0].html
    from crawl4ai.utils import preprocess_html_for_schema
    processed_html = preprocess_html_for_schema(raw_html)
    return JSONResponse({"html": processed_html, "url": body.url, "success": True})

# Screenshot endpoint


@app.post("/screenshot")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool(
    "screenshot",
    title="Website Screenshot Generator",
    readOnlyHint=True,
    openWorldHint=True,
    idempotentHint=True
)
async def generate_screenshot(
    request: Request,
    body: ScreenshotRequest,
    _td: Dict = Depends(token_dep),
):
    """
    📸 **Website Screenshot & Visual Capture Tool**
    
    Captures high-quality, full-page PNG screenshots of any webpage with precise timing control.
    Perfect for visual testing, design analysis, documentation, and automated monitoring.
    
    **📋 CORE FUNCTIONALITY:**
    • Takes full-page screenshots (entire scrollable content, not just viewport)
    • Waits for page load completion and optional custom delays
    • Supports custom timing for dynamic content loading
    • Returns base64-encoded PNG data or saves to specified file path
    • Handles responsive layouts and high-DPI displays
    
    **🎯 PRIMARY USE CASES:**
    
    🖼️ **Visual Documentation**: Capture page layouts for documentation
    🧪 **Visual Testing**: Compare page renders across different configurations  
    📊 **Design Analysis**: Study layout, colors, and visual hierarchy
    🔍 **QA & Monitoring**: Automated visual regression testing
    📱 **Responsive Testing**: Capture mobile vs desktop layouts
    🎨 **Portfolio Creation**: Generate website previews and thumbnails
    
    **⚙️ CONFIGURATION OPTIONS:**
    
    **screenshot_wait_for** (seconds) - Additional delay before capture:
    • 0.0: Capture immediately after page load
    • 2.0: Wait 2 seconds for animations/dynamic content
    • 5.0: Extended wait for slow-loading elements
    
    **output_path** (optional) - Where to save the screenshot:
    • If provided: Returns {"success": true, "path": "/absolute/path/to/file.png"}
    • If omitted: Returns {"success": true, "screenshot": "base64_data"}
    
    **browser_config** - Controls visual output:
    • headless: true/false (false shows actual browser during capture)
    • viewport: {width: 1920, height: 1080} (affects responsive layout)
    • device_scale_factor: 2 (for high-DPI/retina displays)
    • user_agent: "mobile agent" (for mobile-specific rendering)
    
    **crawler_config** - Controls page behavior:
    • wait_for: "css:.page-loaded" (wait for specific elements)
    • page_timeout: 60000 (max time to wait for page load)
    • cache_mode: "bypass" (get fresh render, not cached)
    
    **📐 RESPONSIVE TESTING EXAMPLES:**
    
    📱 **Mobile Screenshot**:
    ```json
    {
      "browser_config": {
        "viewport": {"width": 375, "height": 667},
        "user_agent": "iPhone Safari"
      }
    }
    ```
    
    🖥️ **Desktop Screenshot**:
    ```json
    {
      "browser_config": {
        "viewport": {"width": 1920, "height": 1080}
      }
    }
    ```
    
    **✅ SUCCESS RESPONSE:**
    • With output_path: {"success": true, "path": "/saved/screenshot.png"}
    • Without output_path: {"success": true, "screenshot": "base64_png_data"}
    
    **💡 PRO TIPS:**
    • Use output_path to avoid large base64 data in API responses
    • Set screenshot_wait_for=3.0 for pages with animations
    • Use headless=false for debugging screenshot timing issues
    • Combine with specific viewport settings for responsive testing
    """
    # Build browser config with defaults and user overrides
    browser_cfg = BrowserConfig()
    if body.browser_config:
        browser_cfg = BrowserConfig.from_kwargs(body.browser_config)

    # Build crawler config with screenshot requirements and user overrides
    crawler_cfg_dict = {
        "screenshot": True,
        "screenshot_wait_for": body.screenshot_wait_for
    }
    if body.crawler_config:
        crawler_cfg_dict.update(body.crawler_config)
    crawler_cfg = CrawlerRunConfig.from_kwargs(crawler_cfg_dict)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        results = await crawler.arun(url=body.url, config=crawler_cfg)
    screenshot_data = results[0].screenshot
    if body.output_path:
        abs_path = os.path.abspath(body.output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            f.write(base64.b64decode(screenshot_data))
        return {"success": True, "path": abs_path}
    return {"success": True, "screenshot": screenshot_data}

# PDF endpoint


@app.post("/pdf")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool(
    "pdf",
    title="Website PDF Generator",
    readOnlyHint=True,
    openWorldHint=True,
    idempotentHint=True
)
async def generate_pdf(
    request: Request,
    body: PDFRequest,
    _td: Dict = Depends(token_dep),
):
    """
    📄 **Website PDF Generator & Document Archival Tool**
    
    Converts any webpage into a high-quality PDF document with print-optimized formatting.
    Perfect for creating printable documents, archival snapshots, and offline content distribution.
    
    **📋 CORE FUNCTIONALITY:**
    • Generates print-quality PDF documents from web content
    • Applies browser print CSS and media queries for optimal formatting
    • Handles multi-page content with proper page breaks
    • Returns base64-encoded PDF data or saves to specified file path
    • Maintains text selectability and link functionality in PDF
    
    **🎯 PRIMARY USE CASES:**
    
    📋 **Document Archival**: Create permanent records of web content
    🖨️ **Print-Ready Reports**: Generate documents for offline reading/printing
    📚 **Content Distribution**: Share web content as portable PDF files
    📄 **Legal Documentation**: Archive web pages for compliance/evidence
    📖 **Research Papers**: Convert web articles to academic-style documents
    💼 **Business Reports**: Create professional documents from web data
    
    **⚙️ CONFIGURATION OPTIONS:**
    
    **output_path** (optional) - Where to save the PDF:
    • If provided: Returns {"success": true, "path": "/absolute/path/to/file.pdf"}
    • If omitted: Returns {"success": true, "pdf": "base64_pdf_data"}
    
    **browser_config** - Controls PDF generation:
    • headless: true/false (false for debugging PDF layout)
    • viewport: {width: 1200, height: 800} (affects content layout)
    • user_agent: "print agent" (some sites serve print-specific CSS)
    
    **crawler_config** - Controls page behavior:
    • wait_for: "css:.content-ready" (ensure all content loads)
    • page_timeout: 90000 (PDFs may take longer to generate)
    • excluded_tags: ["nav", "footer"] (clean up content for print)
    • cache_mode: "bypass" (get fresh content for archival)
    
    **📐 PDF FORMATTING EXAMPLES:**
    
    📊 **Report-Style PDF**:
    ```json
    {
      "browser_config": {
        "viewport": {"width": 1200, "height": 1600}
      },
      "crawler_config": {
        "excluded_tags": ["nav", "sidebar", "ads", "footer"],
        "wait_for": "css:.main-content"
      }
    }
    ```
    
    📰 **Article PDF**:
    ```json
    {
      "browser_config": {
        "viewport": {"width": 800, "height": 1200}
      },
      "crawler_config": {
        "excluded_tags": ["header", "nav", "aside", "footer"],
        "css_selector": ".article-content, .post-content"
      }
    }
    ```
    
    **🖨️ PDF QUALITY FEATURES:**
    • Respects CSS @media print rules for optimized formatting
    • Maintains hyperlinks as clickable elements in PDF
    • Preserves images with appropriate resolution
    • Handles page breaks intelligently for readability
    • Includes proper margins and typography for print
    
    **✅ SUCCESS RESPONSE:**
    • With output_path: {"success": true, "path": "/saved/document.pdf"}
    • Without output_path: {"success": true, "pdf": "base64_pdf_data"}
    
    **💡 PRO TIPS:**
    • Use output_path to avoid large base64 data in API responses
    • Exclude navigation elements for cleaner PDF documents
    • Wait for dynamic content to load before PDF generation
    • Use wider viewports (1200px+) for better print formatting
    • Test with headless=false to debug PDF layout issues
    """
    # Build browser config with defaults and user overrides
    browser_cfg = BrowserConfig()
    if body.browser_config:
        browser_cfg = BrowserConfig.from_kwargs(body.browser_config)

    # Build crawler config with PDF requirements and user overrides
    crawler_cfg_dict = {"pdf": True}
    if body.crawler_config:
        crawler_cfg_dict.update(body.crawler_config)
    crawler_cfg = CrawlerRunConfig.from_kwargs(crawler_cfg_dict)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        results = await crawler.arun(url=body.url, config=crawler_cfg)
    pdf_data = results[0].pdf
    if body.output_path:
        abs_path = os.path.abspath(body.output_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as f:
            f.write(pdf_data)
        return {"success": True, "path": abs_path}
    return {"success": True, "pdf": base64.b64encode(pdf_data).decode()}


@app.post("/execute_js")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool(
    "execute_js",
    title="JavaScript Executor",
    readOnlyHint=False,
    openWorldHint=True,
    destructiveHint=True
)
async def execute_js(
    request: Request,
    body: JSEndpointRequest,
    _td: Dict = Depends(token_dep),
):
    """
    Execute a sequence of JavaScript snippets on the specified URL.
    Return the full CrawlResult JSON (first result).
    Use this when you need to interact with dynamic pages using JS.
    REMEMBER: Scripts accept a list of separated JS snippets to execute and execute them in order.
    IMPORTANT: Each script should be an expression that returns a value. It can be an IIFE or an async function. You can think of it as such.
        Your script will replace '{script}' and execute in the browser context. So provide either an IIFE or a sync/async function that returns a value.
    Return Format:
        - The return result is an instance of CrawlResult, so you have access to markdown, links, and other stuff. If this is enough, you don't need to call again for other endpoints.

        ```python
        class CrawlResult(BaseModel):
            url: str
            html: str
            success: bool
            cleaned_html: Optional[str] = None
            media: Dict[str, List[Dict]] = {}
            links: Dict[str, List[Dict]] = {}
            downloaded_files: Optional[List[str]] = None
            js_execution_result: Optional[Dict[str, Any]] = None
            screenshot: Optional[str] = None
            pdf: Optional[bytes] = None
            mhtml: Optional[str] = None
            _markdown: Optional[MarkdownGenerationResult] = PrivateAttr(default=None)
            extracted_content: Optional[str] = None
            metadata: Optional[dict] = None
            error_message: Optional[str] = None
            session_id: Optional[str] = None
            response_headers: Optional[dict] = None
            status_code: Optional[int] = None
            ssl_certificate: Optional[SSLCertificate] = None
            dispatch_result: Optional[DispatchResult] = None
            redirected_url: Optional[str] = None
            network_requests: Optional[List[Dict[str, Any]]] = None
            console_messages: Optional[List[Dict[str, Any]]] = None

        class MarkdownGenerationResult(BaseModel):
            raw_markdown: str
            markdown_with_citations: str
            references_markdown: str
            fit_markdown: Optional[str] = None
            fit_html: Optional[str] = None
        ```

    """
    # Build browser config with defaults and user overrides
    browser_cfg = BrowserConfig()
    if body.browser_config:
        browser_cfg = BrowserConfig.from_kwargs(body.browser_config)

    # Build crawler config with JS requirements and user overrides
    crawler_cfg_dict = {"js_code": body.scripts}
    if body.crawler_config:
        crawler_cfg_dict.update(body.crawler_config)
    crawler_cfg = CrawlerRunConfig.from_kwargs(crawler_cfg_dict)

    async with AsyncWebCrawler(config=browser_cfg) as crawler:
        results = await crawler.arun(url=body.url, config=crawler_cfg)
    # Return JSON-serializable dict of the first CrawlResult
    data = results[0].model_dump()
    return JSONResponse(data)


@app.get("/llm/{url:path}")
async def llm_endpoint(
    request: Request,
    url: str = Path(...),
    q: str = Query(...),
    _td: Dict = Depends(token_dep),
):
    if not q:
        raise HTTPException(400, "Query parameter 'q' is required")
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    answer = await handle_llm_qa(url, q, config)
    return JSONResponse({"answer": answer})


@app.get("/schema")
async def get_schema():
    from crawl4ai import BrowserConfig, CrawlerRunConfig
    return {"browser": BrowserConfig().dump(),
            "crawler": CrawlerRunConfig().dump()}


@app.get(config["observability"]["health_check"]["endpoint"])
async def health():
    return {"status": "ok", "timestamp": time.time(), "version": __version__}


@app.get(config["observability"]["prometheus"]["endpoint"])
async def metrics():
    return RedirectResponse(config["observability"]["prometheus"]["endpoint"])


@app.post("/crawl")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool(
    "crawl",
    title="Multi-URL Web Crawler",
    readOnlyHint=True,
    openWorldHint=True,
    idempotentHint=True
)
async def crawl(
    request: Request,
    crawl_request: CrawlRequest,
    _td: Dict = Depends(token_dep),
):
    """
    🕷️ **Multi-URL Web Crawler & Batch Processor**
    
    Efficiently crawls multiple URLs in parallel and returns comprehensive results for each page.
    Perfect for bulk content extraction, site analysis, competitive research, and data collection at scale.
    
    **📋 CORE FUNCTIONALITY:**
    • Processes multiple URLs simultaneously for faster execution
    • Returns complete CrawlResult objects with all extracted data
    • Handles failures gracefully with detailed error reporting per URL
    • Supports consistent browser and crawler configuration across all URLs
    • Provides structured JSON output for easy integration with data pipelines
    
    **🎯 PRIMARY USE CASES:**
    
    📊 **Competitive Analysis**: Crawl competitor websites for content comparison
    🔍 **Site Auditing**: Analyze multiple pages for SEO, performance, or content issues
    📈 **Market Research**: Collect data from multiple sources for analysis
    🏗️ **Data Pipeline Input**: Bulk content extraction for downstream processing
    📚 **Documentation Crawling**: Extract content from multiple documentation pages
    🌐 **Site Migration**: Backup or migrate content from multiple pages
    
    **📝 INPUT FORMAT:**
    
    **urls** (required) - Array of URLs to crawl:
    ```json
    {
      "urls": [
        "https://example.com/page1",
        "https://example.com/page2", 
        "https://docs.site.com/guide"
      ]
    }
    ```
    
    **⚙️ CONFIGURATION OPTIONS:**
    
    **browser_config** (optional) - Applied to all URLs:
    • headless: true/false (browser visibility)
    • viewport: {width: 1920, height: 1080} (consistent viewport)
    • user_agent: "custom agent" (consistent user agent)
    • proxy: "http://proxy:8080" (use same proxy for all)
    
    **crawler_config** (optional) - Applied to all URLs:
    • wait_for: "css:.content-loaded" (wait condition for all pages)
    • page_timeout: 60000 (timeout for each URL)
    • excluded_tags: ["nav", "footer"] (consistent content filtering)
    • cache_mode: "enabled/disabled/bypass" (caching strategy)
    
    **🚀 PERFORMANCE FEATURES:**
    • **Parallel Processing**: All URLs crawled simultaneously for speed
    • **Individual Error Handling**: Failed URLs don't stop processing of others
    • **Resource Pooling**: Efficient browser instance management
    • **Consistent Configuration**: Same settings applied across all URLs
    
    **📋 COMPLETE CRAWL RESULT DATA:**
    
    Each URL returns a comprehensive CrawlResult object containing:
    ```json
    {
      "url": "original_url",
      "html": "full_page_html",
      "success": true/false,
      "cleaned_html": "processed_html",
      "media": {"images": [...], "videos": [...]},
      "links": {"internal": [...], "external": [...]},
      "markdown": "extracted_markdown_content", 
      "metadata": {"title": "...", "description": "..."},
      "screenshot": "base64_if_requested",
      "pdf": "pdf_data_if_requested",
      "js_execution_result": {...},
      "network_requests": [...],
      "console_messages": [...],
      "status_code": 200,
      "response_headers": {...}
    }
    ```
    
    **💡 BULK PROCESSING EXAMPLES:**
    
    📚 **Documentation Crawling**:
    ```json
    {
      "urls": [
        "https://docs.api.com/getting-started",
        "https://docs.api.com/authentication", 
        "https://docs.api.com/endpoints"
      ],
      "crawler_config": {
        "excluded_tags": ["nav", "sidebar", "footer"],
        "css_selector": ".documentation-content"
      }
    }
    ```
    
    🔍 **Competitive Analysis**:
    ```json
    {
      "urls": [
        "https://competitor1.com/pricing",
        "https://competitor2.com/pricing",
        "https://competitor3.com/pricing"
      ],
      "crawler_config": {
        "wait_for": "css:.pricing-table",
        "excluded_tags": ["header", "footer", "nav"]
      }
    }
    ```
    
    **✅ SUCCESS RESPONSE:**
    Returns JSON array with CrawlResult object for each URL, maintaining input order
    
    **❌ ERROR HANDLING:**
    • Individual URL failures don't stop batch processing
    • Each result includes success/error status and detailed error messages
    • Network timeouts handled gracefully with retry logic
    • Invalid URLs reported with specific error details
    
    **💡 PRO TIPS:**
    • Limit batch size to 10-20 URLs for optimal performance
    • Use consistent browser_config for comparable results across URLs
    • Set appropriate timeouts based on expected page complexity
    • Consider using cache_mode="bypass" for fresh data collection
    """
    if not crawl_request.urls:
        raise HTTPException(400, "At least one URL required")
    res = await handle_crawl_request(
        urls=crawl_request.urls,
        browser_config=crawl_request.browser_config,
        crawler_config=crawl_request.crawler_config,
        config=config,
    )
    return JSONResponse(res)


@app.post("/crawl/stream")
@limiter.limit(config["rate_limiting"]["default_limit"])
async def crawl_stream(
    request: Request,
    crawl_request: CrawlRequest,
    _td: Dict = Depends(token_dep),
):
    if not crawl_request.urls:
        raise HTTPException(400, "At least one URL required")
    crawler, gen = await handle_stream_crawl_request(
        urls=crawl_request.urls,
        browser_config=crawl_request.browser_config,
        crawler_config=crawl_request.crawler_config,
        config=config,
    )
    return StreamingResponse(
        stream_results(crawler, gen),
        media_type="application/x-ndjson",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Stream-Status": "active",
        },
    )


def chunk_code_functions(code_md: str) -> List[str]:
    """Extract each function/class from markdown code blocks per file."""
    pattern = re.compile(
        # match "## File: <path>" then a ```py fence, then capture until the closing ```
        r'##\s*File:\s*(?P<path>.+?)\s*?\r?\n'      # file header
        r'```py\s*?\r?\n'                         # opening fence
        r'(?P<code>.*?)(?=\r?\n```)',             # code block
        re.DOTALL
    )
    chunks: List[str] = []
    for m in pattern.finditer(code_md):
        file_path = m.group("path").strip()
        code_blk = m.group("code")
        tree = ast.parse(code_blk)
        lines = code_blk.splitlines()
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno - 1
                end = getattr(node, "end_lineno", start + 1)
                snippet = "\n".join(lines[start:end])
                chunks.append(f"# File: {file_path}\n{snippet}")
    return chunks


def chunk_doc_sections(doc: str) -> List[str]:
    lines = doc.splitlines(keepends=True)
    sections = []
    current: List[str] = []
    for line in lines:
        if re.match(r"^#{1,6}\s", line):
            if current:
                sections.append("".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append("".join(current))
    return sections


@app.get("/ask")
@limiter.limit(config["rate_limiting"]["default_limit"])
@mcp_tool(
    "ask",
    title="Crawl4AI Knowledge Assistant",
    readOnlyHint=True,
    openWorldHint=False,
    idempotentHint=True
)
async def get_context(
    request: Request,
    _td: Dict = Depends(token_dep),
    context_type: str = Query(
        "all",
        regex="^(code|doc|all)$",
        description="Type of context to retrieve: 'code' for source code, 'doc' for documentation, 'all' for both",
        examples=["all", "doc", "code"]
    ),
    query: Optional[str] = Query(
        None, 
        description="Search query to filter content using BM25 algorithm",
        examples=["async crawling", "browser configuration", "extraction strategies"]
    ),
    score_ratio: float = Query(
        0.5, 
        ge=0.0, 
        le=1.0, 
        description="Minimum score as fraction of maximum score for filtering results",
        examples=[0.3, 0.5, 0.8]
    ),
    max_results: int = Query(
        20, 
        ge=1, 
        le=100,
        description="Maximum number of results to return",
        examples=[5, 10, 20, 50]
    ),
):
    """
    This end point is design for any questions about Crawl4ai library. It returns a plain text markdown with extensive information about Crawl4ai. 
    You can use this as a context for any AI assistant. Use this endpoint for AI assistants to retrieve library context for decision making or code generation tasks.
    Alway is BEST practice you provide a query to filter the context. Otherwise the lenght of the response will be very long.

    Parameters:
    - context_type: Specify "code" for code context, "doc" for documentation context, or "all" for both.
    - query: RECOMMENDED search query to filter paragraphs using BM25. You can leave this empty to get all the context.
    - score_ratio: Minimum score as a fraction of the maximum score for filtering results.
    - max_results: Maximum number of results to return. Default is 20.

    Returns:
    - JSON response with the requested context.
    - If "code" is specified, returns the code context.
    - If "doc" is specified, returns the documentation context.
    - If "all" is specified, returns both code and documentation contexts.
    """
    # load contexts
    base = os.path.dirname(__file__)
    code_path = os.path.join(base, "c4ai-code-context.md")
    doc_path = os.path.join(base, "c4ai-doc-context.md")
    if not os.path.exists(code_path) or not os.path.exists(doc_path):
        raise HTTPException(404, "Context files not found")

    with open(code_path, "r") as f:
        code_content = f.read()
    with open(doc_path, "r") as f:
        doc_content = f.read()

    # if no query, just return raw contexts
    if not query:
        if context_type == "code":
            return JSONResponse({"code_context": code_content})
        if context_type == "doc":
            return JSONResponse({"doc_context": doc_content})
        return JSONResponse({
            "code_context": code_content,
            "doc_context": doc_content,
        })

    tokens = query.split()
    results: Dict[str, List[Dict[str, float]]] = {}

    # code BM25 over functions/classes
    if context_type in ("code", "all"):
        code_chunks = chunk_code_functions(code_content)
        bm25 = BM25Okapi([c.split() for c in code_chunks])
        scores = bm25.get_scores(tokens)
        max_sc = float(scores.max()) if scores.size > 0 else 0.0
        cutoff = max_sc * score_ratio
        picked = [(c, s) for c, s in zip(code_chunks, scores) if s >= cutoff]
        picked = sorted(picked, key=lambda x: x[1], reverse=True)[:max_results]
        results["code_results"] = [{"text": c, "score": s} for c, s in picked]

    # doc BM25 over markdown sections
    if context_type in ("doc", "all"):
        sections = chunk_doc_sections(doc_content)
        bm25d = BM25Okapi([sec.split() for sec in sections])
        scores_d = bm25d.get_scores(tokens)
        max_sd = float(scores_d.max()) if scores_d.size > 0 else 0.0
        cutoff_d = max_sd * score_ratio
        idxs = [i for i, s in enumerate(scores_d) if s >= cutoff_d]
        neighbors = set(i for idx in idxs for i in (idx-1, idx, idx+1))
        valid = [i for i in sorted(neighbors) if 0 <= i < len(sections)]
        valid = valid[:max_results]
        results["doc_results"] = [
            {"text": sections[i], "score": scores_d[i]} for i in valid
        ]

    return JSONResponse(results)


@mcp_resource("config_guide")
def get_config_guide():
    """
    Comprehensive guide for configuring Crawl4AI browser and crawler settings.
    Provides examples and best practices for all MCP tools.
    """
    guide = {
        "title": "Crawl4AI Configuration Guide",
        "description": "Complete guide for browser_config and crawler_config parameters",
        "browser_config": {
            "description": "Browser configuration controls how the browser instance behaves",
            "common_options": {
                "headless": {
                    "type": "boolean",
                    "default": True,
                    "description": "Run browser without GUI. Set to false for debugging.",
                    "examples": [True, False]
                },
                "viewport": {
                    "type": "object",
                    "description": "Browser window dimensions",
                    "examples": [
                        {"width": 1920, "height": 1080},
                        {"width": 1366, "height": 768},
                        {"width": 375, "height": 667}  # Mobile
                    ]
                },
                "user_agent": {
                    "type": "string",
                    "description": "Custom user agent string",
                    "examples": [
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X)"
                    ]
                },
                "proxy": {
                    "type": "string",
                    "description": "Proxy server URL",
                    "examples": [
                        "http://proxy.example.com:8080",
                        "socks5://user:pass@proxy:1080"
                    ]
                },
                "java_script_enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable/disable JavaScript execution",
                    "examples": [True, False]
                }
            }
        },
        "crawler_config": {
            "description": "Crawler configuration controls crawling behavior and content processing",
            "common_options": {
                "cache_mode": {
                    "type": "string",
                    "default": "enabled",
                    "description": "Caching strategy for crawled content",
                    "examples": ["enabled", "disabled", "bypass", "read_only", "write_only"]
                },
                "wait_for": {
                    "type": "string",
                    "description": "CSS selector or condition to wait for before processing",
                    "examples": [
                        "css:.main-content",
                        "css:#dynamic-data",
                        "css:.loading-complete"
                    ]
                },
                "page_timeout": {
                    "type": "number",
                    "default": 60000,
                    "description": "Maximum time to wait for page load (milliseconds)",
                    "examples": [30000, 60000, 120000]
                },
                "excluded_tags": {
                    "type": "array",
                    "description": "HTML tags to remove during processing",
                    "examples": [
                        ["script", "style"],
                        ["nav", "footer", "aside"],
                        ["ads", "banner", "popup"]
                    ]
                },
                "extraction_strategy": {
                    "type": "string",
                    "description": "Content extraction strategy",
                    "examples": [
                        "NoExtractionStrategy",
                        "LLMExtractionStrategy",
                        "CosineStrategy"
                    ]
                },
                "word_count_threshold": {
                    "type": "number",
                    "default": 10,
                    "description": "Minimum words required for content blocks",
                    "examples": [5, 10, 25]
                },
                "only_text": {
                    "type": "boolean",
                    "default": False,
                    "description": "Extract only text content, remove HTML formatting",
                    "examples": [True, False]
                }
            }
        },
        "tool_specific_examples": {
            "md": {
                "description": "Markdown extraction with custom filtering",
                "examples": [
                    {
                        "url": "https://docs.python.org",
                        "filter_type": "fit",
                        "browser_config": {"headless": True, "viewport": {"width": 1920, "height": 1080}},
                        "crawler_config": {"wait_for": "css:.main-content", "excluded_tags": ["nav", "footer"]}
                    }
                ]
            },
            "screenshot": {
                "description": "Screenshot capture with custom viewport",
                "examples": [
                    {
                        "url": "https://example.com",
                        "browser_config": {"headless": False, "viewport": {"width": 1920, "height": 1080}},
                        "crawler_config": {"wait_for": "css:.page-loaded", "screenshot_wait_for": 3.0}
                    }
                ]
            },
            "execute_js": {
                "description": "JavaScript execution with session management",
                "examples": [
                    {
                        "url": "https://app.example.com",
                        "scripts": ["document.title", "localStorage.getItem('user')"],
                        "browser_config": {"headless": False, "java_script_enabled": True},
                        "crawler_config": {"session_id": "js-session", "wait_for": "css:.app-ready"}
                    }
                ]
            }
        },
        "best_practices": [
            "Use headless=False for debugging and development",
            "Set appropriate viewport dimensions for responsive testing",
            "Use wait_for to ensure dynamic content loads before processing",
            "Configure cache_mode based on content freshness requirements",
            "Exclude unnecessary tags to improve processing speed",
            "Set reasonable timeouts based on page complexity"
        ]
    }
    return guide


@mcp_resource("filter_guide")
def get_filter_guide():
    """
    Detailed guide for content filtering strategies and FIT markdown processing.
    Explains how to use different filter types effectively.
    """
    filter_guide = {
        "title": "Crawl4AI Content Filtering Guide",
        "description": "Understanding filter types and FIT markdown for optimal content extraction",
        "filter_types": {
            "raw": {
                "description": "Extract all content without any filtering",
                "use_cases": [
                    "When you need complete page content",
                    "For comprehensive data collection",
                    "When building custom filtering logic"
                ],
                "example": {
                    "filter_type": "raw",
                    "query": None,
                    "result": "Complete HTML converted to markdown with no filtering"
                }
            },
            "fit": {
                "description": "Smart content filtering for readability (FIT = Filter, Identify, Transform)",
                "use_cases": [
                    "Extract main article content",
                    "Remove navigation, ads, and sidebars",
                    "Get clean, readable content"
                ],
                "how_it_works": [
                    "Analyzes page structure and content density",
                    "Identifies main content areas using ML algorithms",
                    "Removes navigation, ads, and boilerplate content",
                    "Optimizes for readability and content quality"
                ],
                "example": {
                    "filter_type": "fit",
                    "query": None,
                    "result": "Clean main content with ads and navigation removed"
                }
            },
            "bm25": {
                "description": "BM25 algorithm-based content filtering with query matching",
                "use_cases": [
                    "Extract content relevant to specific topics",
                    "Search-based content filtering",
                    "When you have specific information needs"
                ],
                "requirements": ["query parameter is required"],
                "example": {
                    "filter_type": "bm25",
                    "query": "installation python setup",
                    "result": "Content sections most relevant to Python installation"
                }
            },
            "llm": {
                "description": "LLM-powered intelligent content filtering",
                "use_cases": [
                    "Complex content understanding",
                    "Context-aware filtering",
                    "When you need AI-powered content selection"
                ],
                "requirements": [
                    "query parameter is required",
                    "LLM configuration must be set up"
                ],
                "example": {
                    "filter_type": "llm",
                    "query": "Extract technical documentation about API usage",
                    "result": "AI-filtered content focused on API documentation"
                }
            }
        },
        "fit_markdown_details": {
            "description": "FIT (Filter, Identify, Transform) is Crawl4AI's intelligent content extraction system",
            "benefits": [
                "Removes ads, navigation, and boilerplate content",
                "Preserves article structure and formatting",
                "Optimizes content for AI processing",
                "Maintains readability and context"
            ],
            "technical_details": [
                "Uses content density analysis",
                "Applies machine learning for content classification",
                "Considers semantic structure of HTML",
                "Balances content quality vs. completeness"
            ]
        },
        "choosing_filters": {
            "raw": "When you need everything and will filter later",
            "fit": "When you want clean, readable main content (recommended for most use cases)",
            "bm25": "When you're looking for specific topics or keywords",
            "llm": "When you need AI-powered understanding of content relevance"
        },
        "advanced_combinations": [
            {
                "scenario": "Documentation extraction",
                "recommendation": {
                    "filter_type": "fit",
                    "crawler_config": {
                        "excluded_tags": ["nav", "footer", "sidebar"],
                        "wait_for": "css:.main-content"
                    }
                }
            },
            {
                "scenario": "Research article processing",
                "recommendation": {
                    "filter_type": "bm25",
                    "query": "methodology results conclusions",
                    "crawler_config": {
                        "word_count_threshold": 25
                    }
                }
            }
        ]
    }
    return filter_guide


# ═══════════════════════════════════════════════════════════════
# MCP PROMPTS - Reusable prompt templates for LLMs
# ═══════════════════════════════════════════════════════════════

@mcp_prompt("browser_config_mobile")
def get_mobile_browser_prompt():
    """
    Prompt template for configuring mobile device simulation in browser_config.
    Helps LLMs understand how to properly configure mobile viewport and user agents.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": """Configure a mobile browser setup for crawling a responsive website. 

I need to simulate an iPhone device and capture content optimized for mobile viewing. Please provide a complete configuration that includes:

1. Mobile viewport dimensions (iPhone-like)
2. Appropriate mobile user agent
3. Device scale factor for retina display
4. Mobile-specific crawler settings

Show me the exact browser_config and crawler_config parameters to use with the Crawl4AI MCP tools."""
            },
            {
                "role": "assistant", 
                "content": """Here's a complete mobile configuration for Crawl4AI:

```json
{
  "url": "https://your-website.com",
  "browser_config": {
    "headless": true,
    "viewport": {
      "width": 375,
      "height": 667,
      "deviceScaleFactor": 2,
      "isMobile": true
    },
    "user_agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
  },
  "crawler_config": {
    "wait_for": "css:.mobile-content-loaded",
    "page_timeout": 45000,
    "excluded_tags": ["script", "noscript"],
    "word_count_threshold": 15
  }
}
```

Key mobile considerations:
- **Viewport**: 375x667 matches iPhone 6/7/8 dimensions
- **deviceScaleFactor**: 2 for retina display simulation
- **User Agent**: Authentic iPhone Safari string
- **Mobile flag**: `isMobile: true` enables touch events
- **Longer timeout**: Mobile networks can be slower
- **Content waiting**: Target mobile-specific CSS classes"""
            }
        ]
    }

@mcp_prompt("browser_config_performance")
def get_performance_browser_prompt():
    """
    Prompt template for configuring browser settings for performance monitoring and testing.
    Helps LLMs set up configurations for measuring page load times and performance metrics.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": """I need to measure website performance and load times using Crawl4AI. Help me configure the browser and crawler settings for accurate performance monitoring.

Requirements:
- Measure actual load times (not cached)
- Capture performance metrics
- Ensure realistic browser conditions
- Handle JavaScript performance timing APIs

What browser_config and crawler_config should I use?"""
            },
            {
                "role": "assistant",
                "content": """Here's a performance monitoring configuration:

```json
{
  "url": "https://target-website.com",
  "browser_config": {
    "headless": true,
    "java_script_enabled": true,
    "light_mode": false,
    "viewport": {"width": 1920, "height": 1080},
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  },
  "crawler_config": {
    "cache_mode": "bypass",
    "page_timeout": 30000,
    "wait_for": "css:body",
    "session_id": "perf-test-session"
  }
}
```

For JavaScript performance measurement, use the `execute_js` tool:

```javascript
// Measure page load performance
const perfData = performance.getEntriesByType('navigation')[0];
const metrics = {
  loadTime: perfData.loadEventEnd - perfData.fetchStart,
  domReady: perfData.domContentLoadedEventEnd - perfData.fetchStart,
  firstByte: perfData.responseStart - perfData.fetchStart,
  domComplete: perfData.domComplete - perfData.fetchStart
};
return metrics;
```

Key performance settings:
- **cache_mode: "bypass"**: Fresh load every time
- **light_mode: false**: Full rendering for accurate timing
- **Minimal wait_for**: Don't add artificial delays
- **JavaScript enabled**: Access to performance APIs"""
            }
        ]
    }

@mcp_prompt("crawler_config_privacy")
def get_privacy_crawler_prompt():
    """
    Prompt template for configuring privacy-focused crawling with proxies and minimal tracking.
    Helps LLMs understand how to set up secure, anonymous crawling configurations.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": """I need to crawl content while maintaining privacy and anonymity. Help me configure Crawl4AI for:

1. Proxy usage (SOCKS5 or HTTP)
2. Minimal browser fingerprinting
3. No caching of sensitive content
4. Text-only extraction for privacy
5. JavaScript disabled for security

Show me the complete configuration."""
            },
            {
                "role": "assistant",
                "content": """Here's a privacy-focused configuration:

```json
{
  "url": "https://sensitive-content.com",
  "browser_config": {
    "headless": true,
    "proxy": "socks5://localhost:9050",
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; rv:91.0) Gecko/20100101 Firefox/91.0",
    "java_script_enabled": false,
    "light_mode": true,
    "text_mode": true,
    "ignore_https_errors": true
  },
  "crawler_config": {
    "cache_mode": "disabled",
    "page_timeout": 120000,
    "excluded_tags": ["script", "style", "img", "video", "audio", "iframe"],
    "only_text": true,
    "extraction_strategy": "NoExtractionStrategy"
  }
}
```

Privacy features explained:
- **SOCKS5 proxy**: Routes through Tor or VPN
- **Minimal user agent**: Generic Firefox string
- **JavaScript disabled**: Prevents tracking scripts
- **Text mode**: Reduces browser fingerprinting
- **Cache disabled**: No local storage of sensitive data
- **Extended timeout**: Proxies can be slower
- **Aggressive filtering**: Remove all media and scripts
- **Text-only extraction**: Maximum privacy

For Tor usage, ensure you have Tor running locally on port 9050."""
            }
        ]
    }

@mcp_prompt("crawler_config_ecommerce")
def get_ecommerce_crawler_prompt():
    """
    Prompt template for configuring e-commerce and product page crawling.
    Helps LLMs set up configurations for extracting product information, prices, and reviews.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": """I want to extract product information from e-commerce websites including prices, descriptions, and availability. 

Help me configure Crawl4AI to:
- Handle dynamic pricing that loads via JavaScript
- Wait for product images to load
- Extract structured product data
- Handle anti-bot measures gracefully
- Focus on main product content

What configuration should I use?"""
            },
            {
                "role": "assistant",
                "content": """Here's an e-commerce optimized configuration:

```json
{
  "url": "https://shop.example.com/product/123",
  "filter_type": "bm25",
  "query": "price cost shipping availability stock description specifications",
  "browser_config": {
    "headless": true,
    "viewport": {"width": 1366, "height": 768},
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "java_script_enabled": true
  },
  "crawler_config": {
    "wait_for": "css:.product-images-loaded, .price-loaded",
    "page_timeout": 60000,
    "excluded_tags": ["nav", "footer", "reviews", "recommendations", "ads"],
    "css_selector": ".product-details, .pricing-info, .product-description",
    "cache_mode": "bypass",
    "word_count_threshold": 10
  }
}
```

For dynamic pricing, use `execute_js` to extract specific data:

```javascript
// Extract dynamic product data
const productData = {
  price: document.querySelector('.price')?.textContent,
  availability: document.querySelector('.availability')?.textContent,
  title: document.querySelector('h1')?.textContent,
  description: document.querySelector('.description')?.textContent,
  images: Array.from(document.querySelectorAll('.product-image img')).map(img => img.src)
};
return productData;
```

E-commerce specific features:
- **BM25 filter**: Target price/product keywords
- **JavaScript enabled**: Dynamic pricing support  
- **Cache bypass**: Always get fresh prices
- **Specific selectors**: Focus on product areas
- **Image wait**: Ensure media loads completely
- **Extended timeout**: Handle slow e-commerce sites"""
            }
        ]
    }

@mcp_prompt("advanced_spa_config")
def get_spa_configuration_prompt():
    """
    Prompt template for configuring Single Page Application (SPA) crawling.
    Helps LLMs understand complex JavaScript application interaction patterns.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": """I need to crawl a complex Single Page Application (SPA) that loads content dynamically. The app has:

- Client-side routing
- Lazy-loaded components  
- Dynamic data fetching
- Authentication requirements
- Complex user interactions needed

Help me configure Crawl4AI to navigate and extract data from this modern web app."""
            },
            {
                "role": "assistant",
                "content": """Here's a comprehensive SPA configuration:

```json
{
  "url": "https://spa-app.com/dashboard",
  "browser_config": {
    "headless": false,
    "java_script_enabled": true,
    "viewport": {"width": 1920, "height": 1080},
    "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
  },
  "crawler_config": {
    "session_id": "spa-session-123",
    "wait_for": "css:[data-testid='app-ready']",
    "page_timeout": 90000,
    "simulation_id": "dashboard-interaction"
  }
}
```

For complex interactions, use `execute_js` with step-by-step navigation:

```javascript
// SPA Navigation and Data Extraction
async function extractSPAData() {
  // Wait for initial app load
  await new Promise(resolve => setTimeout(resolve, 3000));
  
  // Navigate to specific section
  const navButton = document.querySelector('[data-testid="reports-tab"]');
  if (navButton) navButton.click();
  
  // Wait for route change and content load
  await new Promise(resolve => setTimeout(resolve, 2000));
  
  // Extract data after navigation
  const data = {
    currentRoute: window.location.hash || window.location.pathname,
    content: document.querySelector('.main-content')?.textContent,
    dataLoaded: !!document.querySelector('.data-table'),
    timestamp: new Date().toISOString()
  };
  
  return data;
}

return await extractSPAData();
```

SPA-specific considerations:
- **Non-headless mode**: Some SPAs detect headless browsers
- **Extended timeouts**: SPAs take time to initialize
- **Session persistence**: Maintain state across requests
- **Dynamic waiting**: Wait for app-specific ready signals
- **Step-by-step interaction**: Handle complex user flows
- **Route awareness**: Track client-side navigation"""
            }
        ]
    }

@mcp_prompt("documentation_extraction")
def get_documentation_extraction_prompt():
    """
    Prompt template for extracting technical documentation and API references.
    Helps LLMs configure optimal settings for clean, structured documentation crawling.
    """
    return {
        "messages": [
            {
                "role": "user",
                "content": """I need to extract technical documentation from API reference sites and developer docs. I want:

- Clean, well-structured markdown output
- Code examples preserved with formatting
- Navigation and sidebar content removed
- Focus on main documentation content
- Proper handling of syntax highlighting

How should I configure Crawl4AI for documentation extraction?"""
            },
            {
                "role": "assistant",
                "content": """Here's an optimal documentation extraction configuration:

```json
{
  "url": "https://api-docs.service.com/reference",
  "filter_type": "fit",
  "query": "API endpoints methods parameters examples documentation",
  "browser_config": {
    "headless": true,
    "viewport": {"width": 1200, "height": 800},
    "user_agent": "Mozilla/5.0 Documentation Bot"
  },
  "crawler_config": {
    "wait_for": "css:.api-reference-loaded, .documentation-content",
    "excluded_tags": ["nav", "footer", "aside", "sidebar", "cookie-banner", "ads"],
    "word_count_threshold": 15,
    "extraction_strategy": "NoExtractionStrategy",
    "css_selector": ".api-documentation, .main-content, .docs-content",
    "preserve_code": true
  }
}
```

For comprehensive site documentation crawling:

```json
{
  "url": "https://docs.framework.com/getting-started",
  "filter_type": "fit",
  "browser_config": {
    "headless": true,
    "viewport": {"width": 1200, "height": 1000}
  },
  "crawler_config": {
    "wait_for": "css:.docs-loaded",
    "excluded_tags": ["nav", "footer", "search", "toc"],
    "css_selector": "main, .content, .documentation",
    "follow_links": true,
    "max_depth": 2,
    "link_pattern": "/docs/"
  }
}
```

Documentation-specific features:
- **FIT filter**: Removes navigation, preserves main content
- **Code preservation**: Maintains syntax highlighting
- **Specific selectors**: Target documentation containers
- **Link following**: Crawl related documentation pages
- **Clean exclusions**: Remove common doc site elements
- **Wider viewport**: Better rendering of code examples
- **Conservative word threshold**: Include short code snippets"""
            }
        ]
    }


# attach MCP layer (adds /mcp/ws, /mcp/sse, /mcp/schema)
print(f"MCP server running on {config['app']['host']}:{config['app']['port']}")
attach_mcp(
    app,
    base_url=f"http://{config['app']['host']}:{config['app']['port']}"
)

# ────────────────────────── cli ──────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=config["app"]["host"],
        port=config["app"]["port"],
        reload=config["app"]["reload"],
        timeout_keep_alive=config["app"]["timeout_keep_alive"],
    )
# ─────────────────────────────────────────────────────────────
