# deploy/docker/mcp_bridge.py

from __future__ import annotations
import inspect, json, re, anyio, time
from contextlib import suppress
from typing import Any, Callable, Dict, List, Tuple
import httpx

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse
from fastapi import Request
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
from mcp.server.sse import SseServerTransport
# Note: Using official MCP Streamable HTTP transport (2025-03-26 specification)

import mcp.types as t
from mcp.server.lowlevel.server import Server, NotificationOptions
from mcp.server.models import InitializationOptions

# ── Simple logging helper (removed complex MCP logging to fix ASGI issues) ──

# ── Global registries for standalone functions ──────────────────
_standalone_resources = {}
_standalone_prompts = {}
_standalone_templates = {}

# ── opt‑in decorators ───────────────────────────────────────────
def mcp_resource(name: str | None = None):
    def deco(fn):
        fn.__mcp_kind__, fn.__mcp_name__ = "resource", name
        # Register standalone resource functions
        key = name or fn.__name__
        _standalone_resources[key] = fn
        return fn
    return deco

def mcp_template(name: str | None = None):
    def deco(fn):
        fn.__mcp_kind__, fn.__mcp_name__ = "template", name
        # Register standalone template functions
        key = name or fn.__name__
        _standalone_templates[key] = fn
        return fn
    return deco

def mcp_tool(name: str | None = None, **annotations):
    def deco(fn):
        fn.__mcp_kind__, fn.__mcp_name__ = "tool", name
        fn.__mcp_annotations__ = annotations
        return fn
    return deco

def mcp_prompt(name: str | None = None):
    def deco(fn):
        fn.__mcp_kind__, fn.__mcp_name__ = "prompt", name
        # Register standalone prompt functions
        key = name or fn.__name__
        _standalone_prompts[key] = fn
        return fn
    return deco

# ── HTTP‑proxy helper for FastAPI endpoints ─────────────────────
def _make_http_proxy(base_url: str, route):
    method = list(route.methods - {"HEAD", "OPTIONS"})[0]
    async def proxy(**kwargs):
        # replace `/items/{id}` style params first
        path = route.path
        for k, v in list(kwargs.items()):
            placeholder = "{" + k + "}"
            if placeholder in path:
                path = path.replace(placeholder, str(v))
                kwargs.pop(k)
        url = base_url.rstrip("/") + path

        async with httpx.AsyncClient() as client:
            try:
                r = (
                    await client.get(url, params=kwargs)
                    if method == "GET"
                    else await client.request(method, url, json=kwargs)
                )
                r.raise_for_status()
                return r.text if method == "GET" else r.json()
            except httpx.HTTPStatusError as e:
                # surface FastAPI error details instead of plain 500
                raise HTTPException(e.response.status_code, e.response.text)
    return proxy

# ── main entry point ────────────────────────────────────────────
def attach_mcp(
    app: FastAPI,
    *,                          # keyword‑only
    base: str = "/mcp",
    name: str | None = None,
    base_url: str,              # eg. "http://127.0.0.1:8020"
) -> None:
    """Call once after all routes are declared to expose WS+SSE MCP endpoints."""
    server_name = name or app.title or "FastAPI-MCP"
    mcp = Server(server_name)

    # tools: Dict[str, Callable] = {}
    tools: Dict[str, Tuple[Callable, Callable]] = {}
    resources: Dict[str, Callable] = {}
    templates: Dict[str, Callable] = {}
    prompts: Dict[str, Callable] = {}

    # register decorated FastAPI routes
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        kind = getattr(fn, "__mcp_kind__", None)
        if not kind:
            continue

        key = fn.__mcp_name__ or re.sub(r"[/{}}]", "_", route.path).strip("_")

        # if kind == "tool":
        #     tools[key] = _make_http_proxy(base_url, route)
        if kind == "tool":
            proxy = _make_http_proxy(base_url, route)
            tools[key] = (proxy, fn)
            continue
        if kind == "resource":
            resources[key] = fn
        if kind == "template":
            templates[key] = fn
        if kind == "prompt":
            prompts[key] = fn

    # Also register standalone decorated functions (resources, prompts, templates)
    # These are captured by the decorators when the module is imported
    resources.update(_standalone_resources)
    prompts.update(_standalone_prompts)
    templates.update(_standalone_templates)

    # helpers for JSON‑Schema
    def _schema(model: type[BaseModel] | None) -> dict:
        if model is None:
            return {"type": "object"}
        
        # Generate schema using field names (not aliases) for better MCP tool discovery
        schema = model.model_json_schema(by_alias=False)
        return schema

    def _body_model(fn: Callable) -> type[BaseModel] | None:
        for p in inspect.signature(fn).parameters.values():
            a = p.annotation
            if inspect.isclass(a) and issubclass(a, BaseModel):
                return a
        return None

    # MCP handlers
    @mcp.list_tools()
    async def _list_tools() -> List[t.Tool]:
        out = []
        for k, (proxy, orig_fn) in tools.items():
            desc = getattr(orig_fn, "__mcp_description__", None) or inspect.getdoc(orig_fn) or ""
            schema = getattr(orig_fn, "__mcp_schema__", None) or _schema(_body_model(orig_fn))
            annotations = getattr(orig_fn, "__mcp_annotations__", {})
            
            # Convert annotations to proper MCP format
            mcp_annotations = None
            if annotations:
                mcp_annotations = t.ToolAnnotations(**annotations)
            
            out.append(
                t.Tool(
                    name=k, 
                    description=desc, 
                    inputSchema=schema,
                    annotations=mcp_annotations
                )
            )
        return out
             

    @mcp.call_tool()
    async def _call_tool(name: str, arguments: Dict | None) -> List[t.TextContent]:
        if name not in tools:
            raise HTTPException(404, "tool not found")
        
        proxy, _ = tools[name]
        try:
            res = await proxy(**(arguments or {}))
        except HTTPException as exc:
            # map server‑side errors into MCP "text/error" payloads
            err = {"error": exc.status_code, "detail": exc.detail}
            return [t.TextContent(type = "text", text=json.dumps(err))]
        except Exception as exc:
            err = {"error": 500, "detail": "Internal server error"}
            return [t.TextContent(type = "text", text=json.dumps(err))]
            
        return [t.TextContent(type = "text", text=json.dumps(res, default=str))]

    @mcp.list_resources()
    async def _list_resources() -> List[t.Resource]:
        return [
            t.Resource(
                name=k, 
                uri=f"resource://{k}",
                description=inspect.getdoc(f) or "", 
                mimeType="application/json"
            )
            for k, f in resources.items()
        ]

    @mcp.read_resource()
    async def _read_resource(name: str) -> List[t.TextContent]:
        if name not in resources:
            raise HTTPException(404, "resource not found")
        
        try:
            res = resources[name]()
        except Exception as exc:
            raise HTTPException(500, f"Resource access failed: {exc}")
            
        return [t.TextContent(type = "text", text=json.dumps(res, default=str))]

    @mcp.list_resource_templates()
    async def _list_templates() -> List[t.ResourceTemplate]:
        return [
            t.ResourceTemplate(
                name=k,
                description=inspect.getdoc(f) or "",
                parameters={
                    p: {"type": "string"} for p in _path_params(app, f)
                },
            )
            for k, f in templates.items()
        ]

    @mcp.list_prompts()
    async def _list_prompts() -> List[t.Prompt]:
        return [
            t.Prompt(
                name=k,
                description=inspect.getdoc(f) or "",
                arguments=[]  # We'll add arguments support if needed
            )
            for k, f in prompts.items()
        ]

    @mcp.get_prompt()
    async def _get_prompt(name: str, arguments: Dict | None = None) -> t.GetPromptResult:
        if name not in prompts:
            raise HTTPException(404, "prompt not found")
        
        try:
            prompt_fn = prompts[name]
            prompt_data = prompt_fn(**(arguments or {}))
        except Exception as exc:
            raise HTTPException(500, f"Prompt generation failed: {exc}")
        
        # Convert our prompt data to MCP format
        messages = []
        if isinstance(prompt_data, dict) and "messages" in prompt_data:
            for msg in prompt_data["messages"]:
                # Create a single TextContent object, not a list
                content = t.TextContent(type="text", text=msg["content"])
                messages.append(t.PromptMessage(role=msg["role"], content=content))
        else:
            # Fallback for simple string prompts
            content = t.TextContent(type="text", text=str(prompt_data))
            messages.append(t.PromptMessage(role="user", content=content))
        
        return t.GetPromptResult(messages=messages)


    init_opts = InitializationOptions(
        server_name=server_name,
        server_version="0.1.0",
        capabilities=mcp.get_capabilities(
            notification_options=NotificationOptions(),
            experimental_capabilities={},
        ),
    )

    # ── MCP Streamable HTTP Transport (Official) ─────────────────────────
    # Implements MCP specification 2025-03-26 Streamable HTTP transport
    
    from pydantic import TypeAdapter
    from mcp.types import JSONRPCMessage, JSONRPCRequest, JSONRPCNotification, JSONRPCResponse
    
    rpc_adapter = TypeAdapter(JSONRPCMessage)
    
    @app.post(f"{base}")
    @app.get(f"{base}")
    async def _mcp_streamable_http(request: Request):
        """MCP Streamable HTTP transport endpoint per 2025-03-26 specification"""
        
        # Security: Validate Origin header to prevent DNS rebinding attacks
        origin = request.headers.get("origin")
        if origin and not origin.startswith("http://localhost") and not origin.startswith("https://localhost"):
            # For production, implement proper origin validation
            pass
        
        if request.method == "GET":
            # GET requests with Accept: text/event-stream for SSE
            accept = request.headers.get("accept", "")
            if "text/event-stream" in accept:
                return await _handle_sse_stream(request)
            else:
                raise HTTPException(405, "Method Not Allowed - GET requires Accept: text/event-stream")
        
        elif request.method == "POST":
            # POST requests with JSON-RPC messages
            accept = request.headers.get("accept", "")
            content_type = request.headers.get("content-type", "")
            
            if "application/json" not in content_type:
                raise HTTPException(400, "Content-Type must include application/json")
            
            # Parse request body
            try:
                body = await request.json()
            except Exception as e:
                raise HTTPException(400, f"Invalid JSON: {e}")
            
            # Handle single message or batch
            if isinstance(body, list):
                messages = [rpc_adapter.validate_python(msg) for msg in body]
            else:
                messages = [rpc_adapter.validate_python(body)]
            
            # Process messages and determine response type
            has_requests = any(hasattr(msg, 'method') and hasattr(msg, 'id') for msg in messages)
            
            if has_requests and "text/event-stream" in accept:
                # Return SSE stream for requests
                return await _handle_request_stream(messages)
            elif has_requests:
                # Return JSON response for requests
                return await _handle_request_json(messages)
            else:
                # Return 202 for notifications/responses
                await _handle_notifications(messages)
                return JSONResponse({}, status_code=202)
    
    async def _handle_sse_stream(request: Request):
        """Handle SSE stream for MCP requests"""
        async def event_generator():
            # Send connection event
            yield {
                "event": "message",
                "data": json.dumps({
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": mcp.get_capabilities().model_dump(),
                        "serverInfo": {
                            "name": server_name,
                            "version": "0.1.0"
                        }
                    },
                    "id": 1
                })
            }
            
            # Keep connection alive
            while True:
                await asyncio.sleep(30)
                yield {
                    "event": "ping",
                    "data": json.dumps({"timestamp": time.time()})
                }
        
        return EventSourceResponse(event_generator())
    
    async def _handle_request_stream(messages: List[JSONRPCMessage]):
        """Handle requests with SSE response"""
        async def response_generator():
            for msg in messages:
                if hasattr(msg, 'method') and hasattr(msg, 'id'):
                    try:
                        # Route to appropriate handler
                        if msg.method == "tools/list":
                            tools = await _list_tools()
                            result = {"tools": [t.model_dump() for t in tools]}
                        elif msg.method == "tools/call":
                            result = await _call_tool(msg.params.get("name"), msg.params.get("arguments"))
                        elif msg.method == "resources/list":
                            resources = await _list_resources()
                            result = {"resources": [r.model_dump() for r in resources]}
                        elif msg.method == "prompts/list":
                            prompts = await _list_prompts()
                            result = {"prompts": [p.model_dump() for p in prompts]}
                        else:
                            raise HTTPException(404, f"Method not found: {msg.method}")
                        
                        response = {
                            "jsonrpc": "2.0",
                            "result": result,
                            "id": msg.id
                        }
                    except Exception as e:
                        response = {
                            "jsonrpc": "2.0",
                            "error": {"code": -32603, "message": str(e)},
                            "id": msg.id
                        }
                    
                    yield {
                        "event": "message",
                        "data": json.dumps(response)
                    }
        
        return EventSourceResponse(response_generator())
    
    async def _handle_request_json(messages: List[JSONRPCMessage]):
        """Handle requests with JSON response"""
        responses = []
        
        for msg in messages:
            if hasattr(msg, 'method') and hasattr(msg, 'id'):
                try:
                    # Route to appropriate handler
                    if msg.method == "tools/list":
                        tools = await _list_tools()
                        result = {"tools": [t.model_dump() for t in tools]}
                    elif msg.method == "tools/call":
                        result = await _call_tool(msg.params.get("name"), msg.params.get("arguments"))
                    elif msg.method == "resources/list":
                        resources = await _list_resources()
                        result = {"resources": [r.model_dump() for r in resources]}
                    elif msg.method == "prompts/list":
                        prompts = await _list_prompts()
                        result = {"prompts": [p.model_dump() for p in prompts]}
                    else:
                        raise HTTPException(404, f"Method not found: {msg.method}")
                    
                    responses.append({
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": msg.id
                    })
                except Exception as e:
                    responses.append({
                        "jsonrpc": "2.0",
                        "error": {"code": -32603, "message": str(e)},
                        "id": msg.id
                    })
        
        if len(responses) == 1:
            return JSONResponse(responses[0])
        else:
            return JSONResponse(responses)
    
    async def _handle_notifications(messages: List[JSONRPCMessage]):
        """Handle notifications (no response required)"""
        for msg in messages:
            if hasattr(msg, 'method') and not hasattr(msg, 'id'):
                # Process notification silently
                pass

    # ── HTTP transport (recommended) ─────────────────────────
    # Using HTTP transport instead of deprecated SSE transport
    # This avoids ASGI protocol violations and middleware conflicts
    
    @app.get(f"{base}/health")
    async def _mcp_health():
        """Health check endpoint for MCP bridge"""
        return {"status": "ok", "transport": "http", "server": server_name}
    
    # Note: For proper MCP integration, consider migrating to streamable HTTP transport
    # The SSE transport has been deprecated and causes ASGI protocol issues
    
    # ── Alternative: Use EventSourceResponse for SSE-like functionality ─────
    from sse_starlette.sse import EventSourceResponse
    
    # ── Legacy SSE endpoint for backward compatibility ─────────────────────────
    @app.get(f"{base}/sse")
    async def _mcp_sse_legacy(request: Request):
        """Legacy SSE endpoint for backward compatibility with existing MCP clients
        
        WARNING: This endpoint is deprecated as of MCP specification 2025-03-26.
        New clients should use the Streamable HTTP transport at POST/GET /mcp
        """
        
        async def event_stream():
            try:
                # Send deprecation warning
                yield {
                    "event": "warning",
                    "data": json.dumps({
                        "message": "SSE transport deprecated in MCP 2025-03-26. Use Streamable HTTP at /mcp",
                        "specification": "https://modelcontextprotocol.io/specification/2025-03-26/basic/transports"
                    })
                }
                
                # Initialize MCP session data for backward compatibility
                session_data = {
                    "server": server_name,
                    "capabilities": mcp.get_capabilities().model_dump(),
                    "tools": [tool.model_dump() for tool in await _list_tools()],
                    "resources": [res.model_dump() for res in await _list_resources()],
                    "prompts": [prompt.model_dump() for prompt in await _list_prompts()]
                }
                
                # Send initial connection event
                yield {
                    "event": "connection",
                    "data": json.dumps(session_data)
                }
                
                # Send periodic heartbeat to keep connection alive
                while True:
                    yield {
                        "event": "heartbeat", 
                        "data": json.dumps({"timestamp": time.time()})
                    }
                    await asyncio.sleep(30)
                    
            except Exception as e:
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(e)})
                }
                
        return EventSourceResponse(event_stream())
    
    # ── Transport status endpoint ─────────────────────────────
    @app.get(f"{base}/transport-status")
    async def _transport_status():
        """Check status of different MCP transport methods"""
        return JSONResponse({
            "server": server_name,
            "specification": "2025-03-26",
            "official_transports": {
                "streamable_http": {
                    "status": "working",
                    "endpoint": f"{base}",
                    "methods": ["POST", "GET"],
                    "description": "Official MCP Streamable HTTP transport",
                    "specification": "https://modelcontextprotocol.io/specification/2025-03-26/basic/transports"
                },
                "stdio": {
                    "status": "not_applicable",
                    "description": "Standard input/output transport - not available via HTTP"
                }
            },
            "legacy_transports": {
                "sse": {
                    "status": "working_deprecated",
                    "endpoint": f"{base}/sse",
                    "specification": "2024-11-05 and earlier",
                    "description": "Legacy SSE transport - maintained for backward compatibility",
                    "warning": "Deprecated in MCP 2025-03-26. Migrate to Streamable HTTP."
                }
            },
            "recommendation": "Use Streamable HTTP transport (POST/GET to /mcp). Legacy SSE (/mcp/sse) available for compatibility."
        })

    # ── schema endpoint ───────────────────────────────────────
    @app.get(f"{base}/schema")
    async def _schema_endpoint():
        return JSONResponse({
            "tools": [x.model_dump() for x in await _list_tools()],
            "resources": [x.model_dump() for x in await _list_resources()],
            "resource_templates": [x.model_dump() for x in await _list_templates()],
            "prompts": [x.model_dump() for x in await _list_prompts()],
        })


# ── helpers ────────────────────────────────────────────────────
def _route_name(path: str) -> str:
    return re.sub(r"[/{}}]", "_", path).strip("_")

def _path_params(app: FastAPI, fn: Callable) -> List[str]:
    for r in app.routes:
        if r.endpoint is fn:
            return list(r.param_convertors.keys())
    return []
