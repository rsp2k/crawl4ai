# MCP SSE ASGI Protocol Error Analysis and Fixes

## Problem Analysis

The Crawl4AI MCP bridge was experiencing an ASGI protocol violation error:

```
AssertionError: Unexpected message: {'type': 'http.response.start', 'status': 200, 'headers': [(b'content-length', b'4'), (b'content-type', 'application/json')]}
```

This error occurs in `starlette/middleware/base.py` at line 169:
```python
assert message["type"] == "http.response.body", f"Unexpected message: {message}"
```

## Root Causes

### 1. **SSE Transport Deprecation**
- The SSE transport in MCP has been deprecated as of specification version 2025-03-26
- New applications should use Streamable HTTP transport instead

### 2. **ASGI Protocol Violation**
- The MCP SSE transport sends `http.response.start` when middleware expects `http.response.body`
- Using `request._send` (private attribute) bypasses FastAPI's middleware stack
- Middleware interference causes protocol assertion failures

### 3. **Middleware Conflicts**
- FastAPI's security headers middleware
- CORS middleware
- Custom middleware in the application stack
- All can interfere with raw ASGI message flow

## Solutions Implemented

### Solution 1: Alternative SSE Implementation (Immediate Fix)
```python
# Using sse-starlette's EventSourceResponse
@app.get(f"{base}/sse")
async def _mcp_sse_alternative(request: Request):
    async def event_stream():
        # ASGI-compliant event streaming
        session_data = {
            "server": server_name,
            "capabilities": mcp.get_capabilities().model_dump(),
            "tools": [tool.model_dump() for tool in await _list_tools()]
        }
        
        yield {
            "event": "connection",
            "data": json.dumps(session_data)
        }
        
        # Periodic heartbeat
        while True:
            yield {
                "event": "heartbeat", 
                "data": json.dumps({"timestamp": time.time()})
            }
            await asyncio.sleep(30)
            
    return EventSourceResponse(event_stream())
```

### Solution 2: HTTP-Based MCP Transport
```python
# Create session endpoint
@app.post("/mcp/session")
async def create_mcp_session():
    session_data = await http_transport.create_session()
    return JSONResponse(session_data)

# Handle MCP requests via HTTP
@app.post("/mcp/session/{session_id}/request")
async def handle_mcp_request(session_id: str, request_data: Dict[str, Any]):
    response_data = await http_transport.handle_mcp_request(session_id, request_data)
    return JSONResponse(response_data)
```

### Solution 3: Custom ASGI Middleware
```python
class CustomASGIMiddleware:
    async def handle_sse_request(self, scope: Scope, receive: Receive, send: Send):
        # Proper ASGI message handling for SSE
        await send({
            "type": "http.response.start",
            "status": 200,
            "headers": [
                [b"content-type", b"text/event-stream"],
                [b"cache-control", b"no-cache"],
                [b"connection", b"keep-alive"],
            ]
        })
        
        # Send events with proper body messages
        while True:
            event_data = f"event: heartbeat\ndata: {json.dumps({'timestamp': time.time()})}\n\n"
            await send({
                "type": "http.response.body",
                "body": event_data.encode(),
                "more_body": True
            })
            await asyncio.sleep(30)
```

## Transport Status Comparison

| Transport | Status | Pros | Cons | Use Case |
|-----------|--------|------|------|----------|
| **WebSocket** | ✅ Working | Bidirectional, Real-time, Production-ready | More complex | **Recommended for production** |
| **SSE (Original)** | ❌ Broken | Simple | ASGI protocol violation | ⚠️ **Avoid** |
| **SSE (Fixed)** | ✅ Working | Simple, Standards-compliant | One-way only | Development & testing |
| **HTTP Session** | ✅ Working | Simple, Stateless | Request/response only | API integration |

## Implementation Status

### Files Modified
1. **`mcp_bridge.py`** - Main bridge implementation with fixes
2. **`mcp_sse_fix.py`** - Dedicated fix implementations
3. **`MCP_SSE_ASGI_FIXES.md`** - This documentation

### New Endpoints Added
- `/mcp/transport-status` - Check transport method status
- `/mcp/sse-fixed` - ASGI-compliant SSE implementation
- `/mcp/session` - HTTP-based session management
- `/mcp/session/{session_id}/request` - HTTP MCP protocol handler
- `/mcp/health` - Health check endpoint

## Testing the Fixes

### 1. Check Transport Status
```bash
curl http://localhost:8000/mcp/transport-status
```

### 2. Test Fixed SSE Endpoint
```bash
curl -H "Accept: text/event-stream" http://localhost:8000/mcp/sse-fixed
```

### 3. Test HTTP Session
```bash
# Create session
curl -X POST http://localhost:8000/mcp/session

# Use session
curl -X POST http://localhost:8000/mcp/session/session_123/request \
  -H "Content-Type: application/json" \
  -d '{"method": "tools/list", "id": 1}'
```

### 4. Test WebSocket (Recommended)
```javascript
const ws = new WebSocket('ws://localhost:8000/mcp/ws');
ws.onopen = () => {
    ws.send(JSON.stringify({
        jsonrpc: '2.0',
        method: 'initialize',
        id: 1,
        params: {}
    }));
};
```

## Recommendations

### For Production
1. **Use WebSocket transport** - Most reliable and feature-complete
2. **Disable original SSE endpoint** - Prevents ASGI protocol errors
3. **Monitor transport status** - Use `/mcp/transport-status` endpoint

### For Development
1. **Use fixed SSE endpoint** - Easier debugging with browser dev tools
2. **Test with MCP Inspector** - Use the fixed SSE endpoint URL
3. **Check logs regularly** - Monitor for any remaining issues

### For Integration
1. **Use HTTP session transport** - Simplest for API clients
2. **Implement proper session management** - Handle session timeouts
3. **Add authentication** - Secure session creation and usage

## Migration Path

### Phase 1: Immediate Fix (Completed)
- ✅ Comment out problematic SSE transport
- ✅ Add alternative SSE implementation  
- ✅ Add transport status monitoring

### Phase 2: Transition Period
- 🔄 Test all transport methods thoroughly
- 🔄 Update client applications to use WebSocket
- 🔄 Monitor error logs for any remaining issues

### Phase 3: Long-term Solution
- 📋 Migrate to Streamable HTTP transport (when available)
- 📋 Remove legacy SSE implementations
- 📋 Optimize WebSocket performance

## Error Prevention

### Code Patterns to Avoid
```python
# ❌ DON'T: Use private ASGI attributes
await mcp.run(read_stream, write_stream, init_opts)
async with sse.connect_sse(request.scope, request.receive, request._send):

# ❌ DON'T: Bypass middleware with raw ASGI
app.mount("/messages", app=sse.handle_post_message)
```

### Code Patterns to Use
```python
# ✅ DO: Use EventSourceResponse for SSE
return EventSourceResponse(event_generator())

# ✅ DO: Handle ASGI messages properly
await send({"type": "http.response.body", "body": data, "more_body": True})

# ✅ DO: Use WebSocket for real-time communication
@app.websocket("/mcp/ws")
async def websocket_endpoint(websocket: WebSocket):
```

## Additional Resources

- [MCP Streamable HTTP Transport Specification](https://spec.modelcontextprotocol.io/specification/transports/http/)
- [FastAPI WebSocket Documentation](https://fastapi.tiangolo.com/advanced/websockets/)
- [SSE-Starlette Documentation](https://github.com/sysid/sse-starlette)
- [ASGI Specification](https://asgi.readthedocs.io/en/latest/specs/main.html)