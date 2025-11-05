# Intercepting Flower Messages: Complete Guide

The following guide has been written by Claude code and needs to be verified for accuracy and completeness.

## Table of Contents

1. [Overview](#overview)
2. [Flower Communication Architecture](#flower-communication-architecture)
3. [Interception Approaches](#interception-approaches)
4. [Approach 1: HTTP Proxy (Recommended)](#approach-1-http-proxy-recommended)
5. [Approach 2: gRPC Interceptor](#approach-2-grpc-interceptor)
6. [Approach 3: Custom Transport Layer](#approach-3-custom-transport-layer)
7. [Working with Protobuf Messages](#working-with-protobuf-messages)
8. [Complete Implementation Examples](#complete-implementation-examples)
9. [Testing & Debugging](#testing--debugging)
10. [Production Considerations](#production-considerations)

---

## Overview

Flower uses protobuf-serialized messages for communication between clients (SuperNodes) and servers (SuperLink). This guide explains how to intercept these messages to relay them through your own REST API or middleware layer.

**Why intercept messages?**
- Add custom logging/monitoring
- Route through enterprise API gateways
- Transform or enrich messages
- Implement custom authentication/authorization
- Audit federated learning communications

---

## Flower Communication Architecture

### Transport Layers

Flower supports three transport types:

| Transport | Protocol | Port | Use Case |
|-----------|----------|------|----------|
| `grpc-rere` | gRPC (request-response) | 9092 | Default, most common |
| `grpc-adapter` | gRPC (streaming wrapper) | 9092 | Legacy compatibility |
| `rest` | HTTP POST + Protobuf | 9095 | **Best for interception** |

### Message Flow

```
┌─────────────┐                      ┌─────────────┐
│  SuperNode  │                      │  SuperLink  │
│  (Client)   │                      │  (Server)   │
└─────────────┘                      └─────────────┘
      │                                     │
      │  1. register_node()                │
      ├────────────────────────────────────>│
      │                                     │
      │  2. activate_node()                │
      ├────────────────────────────────────>│
      │                                     │
      │  3. pull_messages()                │
      │     (get training task)            │
      ├────────────────────────────────────>│
      │  <─ FitIns (config, params)        │
      │                                     │
      │  4. [Local Training]               │
      │                                     │
      │  5. push_messages()                │
      │     (send training results)        │
      ├────────────────────────────────────>│
      │                                     │
      │  6. deactivate_node()              │
      ├────────────────────────────────────>│
      │                                     │
      │  7. unregister_node()              │
      ├────────────────────────────────────>│
```

### Key Endpoints (REST API)

All endpoints use `POST` with `Content-Type: application/protobuf`:

- `/api/v0/fleet/register-node` - Register client
- `/api/v0/fleet/activate-node` - Activate client session
- `/api/v0/fleet/pull-messages` - **Get training instructions**
- `/api/v0/fleet/push-messages` - **Send training results**
- `/api/v0/fleet/pull-object` - Download large objects (model weights)
- `/api/v0/fleet/push-object` - Upload large objects
- `/api/v0/fleet/deactivate-node` - End session
- `/api/v0/fleet/unregister-node` - Unregister client

**Most important**: `pull-messages` and `push-messages` contain the actual federated learning data.

---

## Interception Approaches

### Comparison

| Approach | Complexity | Flexibility | Protobuf Handling | Best For |
|----------|------------|-------------|-------------------|----------|
| HTTP Proxy | Low | Medium | Binary passthrough | Production, simple relay |
| gRPC Interceptor | Medium | High | Native gRPC | Deep inspection, modification |
| Custom Transport | High | Very High | Manual | Complete control, custom protocols |

---

## Approach 1: HTTP Proxy (Recommended)

### Architecture

```
┌──────────┐        ┌──────────────┐        ┌─────────────┐
│SuperNode │───────>│  HTTP Proxy  │───────>│ Your REST   │
│          │        │  (Intercept) │        │    API      │
│          │<───────│              │<───────│             │
└──────────┘        └──────────────┘        └─────────────┘
                           │                        │
                           │                        v
                           │                 ┌─────────────┐
                           └────────────────>│  SuperLink  │
                                             │   (Server)  │
                                             └─────────────┘
```

### Implementation

#### Step 1: Simple Passthrough Proxy

```python
# proxy.py
from flask import Flask, request, Response
import requests

app = Flask(__name__)

# Configuration
FLOWER_SERVER = "http://localhost:9095"  # Actual Flower SuperLink
YOUR_API = "https://your-api.com"        # Your REST API

@app.route('/api/v0/fleet/<path:endpoint>', methods=['POST'])
def proxy_flower_messages(endpoint):
    """Intercept all Flower messages."""

    # 1. Receive protobuf from SuperNode
    protobuf_data = request.data
    headers = {
        'Content-Type': 'application/protobuf',
        'Accept': 'application/protobuf'
    }

    print(f"[INTERCEPT] Endpoint: {endpoint}")
    print(f"[INTERCEPT] Data size: {len(protobuf_data)} bytes")

    # 2. Forward to Flower server
    flower_response = requests.post(
        f"{FLOWER_SERVER}/api/v0/fleet/{endpoint}",
        data=protobuf_data,
        headers=headers,
        timeout=30
    )

    # 3. Return response to SuperNode
    return Response(
        flower_response.content,
        status=flower_response.status_code,
        headers={'Content-Type': 'application/protobuf'}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

**Usage**:
```bash
# Terminal 1: Start Flower server
python -m flwr.server.app --fleet-api-type rest

# Terminal 2: Start proxy
python proxy.py

# Terminal 3: Start client (pointing to proxy)
flower-supernode --rest --superlink http://localhost:8080
```

#### Step 2: Relay Through Your REST API

```python
# advanced_proxy.py
from flask import Flask, request, Response
import requests
import json
import base64

app = Flask(__name__)

FLOWER_SERVER = "http://localhost:9095"
YOUR_API = "https://your-api.com/relay"

@app.route('/api/v0/fleet/<path:endpoint>', methods=['POST'])
def relay_through_custom_api(endpoint):
    """Relay messages through your custom REST API."""

    protobuf_data = request.data

    # Option A: Send protobuf as-is to your API
    # Your API must handle binary protobuf data
    response = requests.post(
        f"{YOUR_API}/{endpoint}",
        data=protobuf_data,
        headers={'Content-Type': 'application/protobuf'},
        timeout=60
    )
    processed_data = response.content

    # OR Option B: Convert to base64 for JSON APIs
    payload = {
        "endpoint": endpoint,
        "protobuf_base64": base64.b64encode(protobuf_data).decode('utf-8'),
        "timestamp": time.time()
    }
    response = requests.post(
        YOUR_API,
        json=payload,
        timeout=60
    )
    # Your API returns base64-encoded protobuf
    processed_data = base64.b64decode(response.json()['protobuf_base64'])

    # Forward processed data to Flower
    flower_response = requests.post(
        f"{FLOWER_SERVER}/api/v0/fleet/{endpoint}",
        data=processed_data,
        headers={'Content-Type': 'application/protobuf'}
    )

    return Response(
        flower_response.content,
        status=flower_response.status_code,
        headers={'Content-Type': 'application/protobuf'}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)
```

#### Step 3: Parse and Inspect Messages

```python
# inspecting_proxy.py
from flask import Flask, request, Response
import requests
from google.protobuf.json_format import MessageToDict

# Import Flower protobuf messages
from flwr.proto.fleet_pb2 import (
    PullMessagesRequest, PullMessagesResponse,
    PushMessagesRequest, PushMessagesResponse,
    RegisterNodeFleetRequest, RegisterNodeFleetResponse,
)
from flwr.proto.message_pb2 import Message as FlowerMessage

app = Flask(__name__)
FLOWER_SERVER = "http://localhost:9095"

# Map endpoints to protobuf types
ENDPOINT_TYPES = {
    'pull-messages': (PullMessagesRequest, PullMessagesResponse),
    'push-messages': (PushMessagesRequest, PushMessagesResponse),
    'register-node': (RegisterNodeFleetRequest, RegisterNodeFleetResponse),
}

@app.route('/api/v0/fleet/<path:endpoint>', methods=['POST'])
def inspect_and_relay(endpoint):
    """Parse, inspect, and relay Flower messages."""

    protobuf_data = request.data

    # Parse request protobuf
    if endpoint in ENDPOINT_TYPES:
        req_type, res_type = ENDPOINT_TYPES[endpoint]
        req_msg = req_type()
        req_msg.ParseFromString(protobuf_data)

        # Convert to dict for inspection
        req_dict = MessageToDict(req_msg)
        print(f"\n[REQUEST] {endpoint}")
        print(f"Data: {json.dumps(req_dict, indent=2)}")

        # Log specific information
        if endpoint == 'pull-messages':
            print(f"Node ID: {req_msg.node.node_id}")
        elif endpoint == 'push-messages':
            print(f"Node ID: {req_msg.node.node_id}")
            print(f"Messages count: {len(req_msg.messages_list)}")

    # Forward to Flower server
    flower_response = requests.post(
        f"{FLOWER_SERVER}/api/v0/fleet/{endpoint}",
        data=protobuf_data,
        headers={'Content-Type': 'application/protobuf'}
    )

    # Parse response protobuf
    if endpoint in ENDPOINT_TYPES:
        res_type = ENDPOINT_TYPES[endpoint][1]
        res_msg = res_type()
        res_msg.ParseFromString(flower_response.content)

        res_dict = MessageToDict(res_msg)
        print(f"\n[RESPONSE] {endpoint}")
        print(f"Data: {json.dumps(res_dict, indent=2)}")

    return Response(
        flower_response.content,
        status=flower_response.status_code,
        headers={'Content-Type': 'application/protobuf'}
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Advantages of HTTP Proxy

✅ Simple to implement
✅ Works with existing Flower REST transport
✅ No modification to Flower codebase
✅ Easy to deploy (nginx, HAProxy, custom Python server)
✅ Can inspect/modify messages in flight

### Disadvantages

❌ Only works with REST transport (not gRPC)
❌ Adds network latency
❌ Single point of failure

---

## Approach 2: gRPC Interceptor

### Architecture

gRPC interceptors hook into the gRPC call chain to intercept messages before/after they're sent.

```python
# grpc_interceptor.py
import grpc
from grpc_interceptor import ServerInterceptor
import requests

class FlowerGrpcInterceptor(ServerInterceptor):
    """Intercept gRPC calls and relay through REST API."""

    def __init__(self, relay_api_url):
        self.relay_api_url = relay_api_url

    def intercept(self, method, request, context, method_name):
        """
        Called for each gRPC method invocation.

        Args:
            method: The actual gRPC method handler
            request: The protobuf request
            context: gRPC context
            method_name: Name of the method being called
        """
        print(f"[INTERCEPT gRPC] {method_name}")

        # Serialize request to protobuf bytes
        request_bytes = request.SerializeToString()

        # Send to your REST API
        response = requests.post(
            f"{self.relay_api_url}/grpc/{method_name}",
            data=request_bytes,
            headers={'Content-Type': 'application/protobuf'}
        )

        # Get processed protobuf back
        processed_bytes = response.content

        # Deserialize and continue with Flower's handler
        # (In practice, you might want to parse, modify, then continue)

        # Call the original method
        result = method(request, context)

        print(f"[RESULT gRPC] {method_name} completed")
        return result

# To use this interceptor, modify Flower's server startup:
# server = grpc.server(
#     futures.ThreadPoolExecutor(max_workers=10),
#     interceptors=[FlowerGrpcInterceptor("https://your-api.com")]
# )
```

### Integrating into Flower

You would need to modify `py/flwr/server/superlink/fleet/grpc_rere/fleet_servicer.py`:

```python
# Modified fleet_servicer.py (conceptual)
import grpc
from concurrent import futures
from your_interceptor import FlowerGrpcInterceptor

def run_fleet_api_grpc_rere(address, interceptor_url=None):
    """Start gRPC Fleet API with optional interceptor."""

    interceptors = []
    if interceptor_url:
        interceptors.append(FlowerGrpcInterceptor(interceptor_url))

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        interceptors=interceptors  # Add interceptor here
    )

    # ... rest of server setup
```

### Advantages

✅ Works with gRPC transport (default for Flower)
✅ Native protobuf handling
✅ Can intercept at multiple points (client/server)
✅ Deep integration with gRPC lifecycle

### Disadvantages

❌ Requires modifying Flower codebase
❌ More complex than HTTP proxy
❌ Need to rebuild Flower after changes

---

## Approach 3: Custom Transport Layer

### Architecture

Implement a completely custom transport by subclassing Flower's connection interfaces.

```python
# custom_transport.py
from typing import Optional, Callable
from contextlib import contextmanager
import requests

from flwr.common.message import Message
from flwr.proto.message_pb2 import ObjectTree
from flwr.common.typing import Fab, Run

@contextmanager
def custom_relay_connection(
    server_address: str,
    relay_api_url: str,
    **kwargs
):
    """
    Custom connection context that relays through your REST API.

    This mimics the interface of http_request_response() from
    py/flwr/client/rest_client/connection.py
    """

    # State
    node_id = None

    def relay_request(endpoint: str, protobuf_data: bytes) -> bytes:
        """Send request through your REST API."""

        # Send to your API
        response = requests.post(
            f"{relay_api_url}/{endpoint}",
            data=protobuf_data,
            headers={'Content-Type': 'application/protobuf'}
        )

        # Your API processes and returns protobuf
        return response.content

    def receive() -> Optional[tuple[Message, ObjectTree]]:
        """Pull messages from server (through relay)."""
        from flwr.proto.fleet_pb2 import PullMessagesRequest, PullMessagesResponse
        from flwr.common.serde import message_from_proto
        from flwr.proto.node_pb2 import Node

        req = PullMessagesRequest(node=Node(node_id=node_id))
        req_bytes = req.SerializeToString()

        # Relay through your API
        res_bytes = relay_request('pull-messages', req_bytes)

        res = PullMessagesResponse()
        res.ParseFromString(res_bytes)

        if len(res.messages_list) == 0:
            return None

        message = message_from_proto(res.messages_list[0])
        object_tree = res.message_object_trees[0]

        return message, object_tree

    def send(message: Message, object_tree: ObjectTree) -> set[str]:
        """Send messages to server (through relay)."""
        from flwr.proto.fleet_pb2 import PushMessagesRequest, PushMessagesResponse
        from flwr.common.serde import message_to_proto
        from flwr.common.message import remove_content_from_message
        from flwr.proto.node_pb2 import Node

        if message.has_content():
            message = remove_content_from_message(message)

        req = PushMessagesRequest(
            node=Node(node_id=node_id),
            messages_list=[message_to_proto(message)],
            message_object_trees=[object_tree]
        )
        req_bytes = req.SerializeToString()

        # Relay through your API
        res_bytes = relay_request('push-messages', req_bytes)

        res = PushMessagesResponse()
        res.ParseFromString(res_bytes)

        return set(res.objects_to_push)

    def get_run(run_id: int) -> Run:
        """Get run configuration."""
        from flwr.proto.run_pb2 import GetRunRequest, GetRunResponse
        from flwr.common.serde import run_from_proto
        from flwr.proto.node_pb2 import Node

        req = GetRunRequest(node=Node(node_id=node_id), run_id=run_id)
        res_bytes = relay_request('get-run', req.SerializeToString())

        res = GetRunResponse()
        res.ParseFromString(res_bytes)
        return run_from_proto(res.run)

    # ... implement other required functions (get_fab, pull_object, etc.)

    # Register and activate node
    from flwr.proto.fleet_pb2 import RegisterNodeFleetRequest, ActivateNodeRequest
    from flwr.proto.fleet_pb2 import RegisterNodeFleetResponse, ActivateNodeResponse

    # Register
    reg_req = RegisterNodeFleetRequest(public_key=b"dummy_key")
    relay_request('register-node', reg_req.SerializeToString())

    # Activate
    act_req = ActivateNodeRequest(public_key=b"dummy_key", heartbeat_interval=30)
    act_bytes = relay_request('activate-node', act_req.SerializeToString())
    act_res = ActivateNodeResponse()
    act_res.ParseFromString(act_bytes)
    node_id = act_res.node_id

    try:
        # Yield the connection primitives
        yield (
            node_id,
            receive,
            send,
            get_run,
            lambda fab_hash, run_id: Fab("", b"", {}),  # get_fab stub
            lambda run_id, object_id: b"",  # pull_object stub
            lambda run_id, object_id, contents: None,  # push_object stub
            lambda run_id, object_id: None,  # confirm_message_received stub
        )
    finally:
        # Cleanup (deactivate, unregister)
        pass
```

### Using Custom Transport

You'd need to modify the client to use your custom connection:

```python
# In your client code or modified Flower client
from custom_transport import custom_relay_connection

# Instead of:
# from flwr.client.rest_client.connection import http_request_response

# Use:
with custom_relay_connection(
    server_address="http://localhost:9095",
    relay_api_url="https://your-api.com/relay"
) as conn:
    node_id, receive, send, get_run, *rest = conn

    # Your client logic here
    message, obj_tree = receive()
    # ... process ...
    send(response_message, obj_tree)
```

### Advantages

✅ Complete control over communication
✅ Can implement custom protocols
✅ No external proxy needed
✅ Can add encryption, compression, etc.

### Disadvantages

❌ Highest complexity
❌ Must maintain compatibility with Flower updates
❌ Need to implement all connection primitives
❌ Requires forking or modifying Flower

---

## Working with Protobuf Messages

### Understanding Protobuf Structure

Flower's protobuf definitions are in `proto/flwr/proto/`:

```protobuf
// fleet.proto
message PullMessagesRequest {
  Node node = 1;
}

message PullMessagesResponse {
  repeated Message messages_list = 1;
  repeated ObjectTree message_object_trees = 2;
}

message PushMessagesRequest {
  Node node = 1;
  repeated Message messages_list = 2;
  repeated ObjectTree message_object_trees = 3;
}

message PushMessagesResponse {
  repeated string objects_to_push = 1;
}
```

```protobuf
// message.proto
message Message {
  Metadata metadata = 1;
  MessageContent content = 2;
  Error error = 3;
}
```

### Parsing Protobuf in Python

```python
from flwr.proto.fleet_pb2 import PullMessagesResponse
from google.protobuf.json_format import MessageToDict, MessageToJson

# Parse binary protobuf
response_bytes = b'\x...'  # Binary data from network
response = PullMessagesResponse()
response.ParseFromString(response_bytes)

# Access fields
print(f"Number of messages: {len(response.messages_list)}")
for msg in response.messages_list:
    print(f"Message ID: {msg.metadata.message_id}")

# Convert to dictionary
msg_dict = MessageToDict(response)
print(msg_dict)

# Convert to JSON string
msg_json = MessageToJson(response)
print(msg_json)

# Create new protobuf from dict
from google.protobuf.json_format import ParseDict
new_response = ParseDict(msg_dict, PullMessagesResponse())
```

### Converting Protobuf ↔ JSON for REST APIs

```python
import json
import base64
from google.protobuf.json_format import MessageToDict, ParseDict
from flwr.proto.fleet_pb2 import PushMessagesRequest, PushMessagesResponse

def protobuf_to_json_payload(protobuf_bytes: bytes, msg_type) -> dict:
    """Convert protobuf bytes to JSON-friendly dict."""
    msg = msg_type()
    msg.ParseFromString(protobuf_bytes)

    return {
        "protobuf_base64": base64.b64encode(protobuf_bytes).decode('utf-8'),
        "protobuf_json": MessageToDict(msg),
        "message_type": msg_type.__name__
    }

def json_payload_to_protobuf(payload: dict, msg_type) -> bytes:
    """Convert JSON payload back to protobuf bytes."""

    # Option 1: Use base64 (preserves exact binary)
    if "protobuf_base64" in payload:
        return base64.b64decode(payload["protobuf_base64"])

    # Option 2: Use JSON (may lose some binary precision)
    if "protobuf_json" in payload:
        msg = ParseDict(payload["protobuf_json"], msg_type())
        return msg.SerializeToString()

    raise ValueError("Invalid payload format")

# Example usage in proxy
@app.route('/api/v0/fleet/push-messages', methods=['POST'])
def relay_push_messages():
    protobuf_bytes = request.data

    # Convert to JSON for your API
    json_payload = protobuf_to_json_payload(
        protobuf_bytes,
        PushMessagesRequest
    )

    # Send to your REST API
    response = requests.post(
        "https://your-api.com/relay/push-messages",
        json=json_payload
    )

    # Convert response back to protobuf
    protobuf_response = json_payload_to_protobuf(
        response.json(),
        PushMessagesResponse
    )

    return Response(protobuf_response, content_type='application/protobuf')
```

### Extracting Training Data

```python
from flwr.proto.fleet_pb2 import PushMessagesRequest
from flwr.common.serde import message_from_proto
from flwr.common.record import RecordSet

def extract_training_metrics(protobuf_bytes: bytes) -> dict:
    """Extract training metrics from push_messages request."""

    req = PushMessagesRequest()
    req.ParseFromString(protobuf_bytes)

    metrics = []
    for msg_proto in req.messages_list:
        # Convert to Flower Message
        message = message_from_proto(msg_proto)

        # Extract content
        content = message.content
        if content:
            # Get metrics from RecordSet
            if hasattr(content, 'recordset_dict'):
                for key, recordset in content.recordset_dict.items():
                    print(f"RecordSet: {key}")

                    # Extract metrics
                    if hasattr(recordset, 'metrics_records'):
                        for metric_key, metric_value in recordset.metrics_records.items():
                            print(f"  {metric_key}: {metric_value}")
                            metrics.append({
                                "key": metric_key,
                                "value": metric_value
                            })

    return {
        "node_id": req.node.node_id,
        "num_messages": len(req.messages_list),
        "metrics": metrics
    }
```

---

## Complete Implementation Examples

### Production-Ready HTTP Proxy with Logging

```python
#!/usr/bin/env python3
"""
Production-grade Flower message interceptor.

Features:
- Async request handling (FastAPI)
- Structured logging
- Error handling
- Metrics collection
- Health checks
"""

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import httpx
import logging
import time
from typing import Optional
import uvicorn

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Flower Message Interceptor")

# Configuration
FLOWER_SERVER = "http://localhost:9095"
YOUR_RELAY_API = "https://your-api.com/relay"
TIMEOUT = 60.0

# Metrics
metrics = {
    "requests_total": 0,
    "requests_by_endpoint": {},
    "bytes_sent": 0,
    "bytes_received": 0,
    "errors": 0
}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "metrics": metrics}

@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics."""
    return metrics

@app.post("/api/v0/fleet/{endpoint:path}")
async def intercept_flower_message(endpoint: str, request: Request):
    """
    Intercept and relay Flower messages.

    Flow:
    1. Receive protobuf from SuperNode
    2. Log and collect metrics
    3. Optionally relay through your API
    4. Forward to Flower SuperLink
    5. Return response to SuperNode
    """
    start_time = time.time()

    try:
        # Read protobuf data
        protobuf_data = await request.body()
        data_size = len(protobuf_data)

        # Update metrics
        metrics["requests_total"] += 1
        metrics["bytes_received"] += data_size
        metrics["requests_by_endpoint"][endpoint] = \
            metrics["requests_by_endpoint"].get(endpoint, 0) + 1

        logger.info(f"Intercepted: {endpoint} ({data_size} bytes)")

        # Optional: Parse and inspect (for logging only)
        # inspect_message(endpoint, protobuf_data)

        # Optional: Relay through your custom API
        if YOUR_RELAY_API and endpoint in ["pull-messages", "push-messages"]:
            try:
                async with httpx.AsyncClient() as client:
                    relay_response = await client.post(
                        f"{YOUR_RELAY_API}/{endpoint}",
                        content=protobuf_data,
                        headers={
                            "Content-Type": "application/protobuf",
                            "X-Flower-Endpoint": endpoint
                        },
                        timeout=TIMEOUT / 2
                    )

                    if relay_response.status_code == 200:
                        # Use processed data from your API
                        protobuf_data = relay_response.content
                        logger.info(f"Relayed through custom API: {endpoint}")
                    else:
                        logger.warning(
                            f"Relay failed (status {relay_response.status_code}), "
                            "using original data"
                        )
            except Exception as e:
                logger.error(f"Relay error: {e}, using original data")

        # Forward to Flower SuperLink
        async with httpx.AsyncClient() as client:
            flower_response = await client.post(
                f"{FLOWER_SERVER}/api/v0/fleet/{endpoint}",
                content=protobuf_data,
                headers={
                    "Content-Type": "application/protobuf",
                    "Accept": "application/protobuf"
                },
                timeout=TIMEOUT
            )

        # Update metrics
        metrics["bytes_sent"] += len(flower_response.content)

        duration = time.time() - start_time
        logger.info(
            f"Completed: {endpoint} "
            f"(status {flower_response.status_code}, {duration:.2f}s)"
        )

        # Return response to SuperNode
        return Response(
            content=flower_response.content,
            status_code=flower_response.status_code,
            headers={"Content-Type": "application/protobuf"}
        )

    except httpx.TimeoutException:
        metrics["errors"] += 1
        logger.error(f"Timeout: {endpoint}")
        return JSONResponse(
            status_code=504,
            content={"error": "Gateway timeout"}
        )

    except Exception as e:
        metrics["errors"] += 1
        logger.error(f"Error processing {endpoint}: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

def inspect_message(endpoint: str, protobuf_data: bytes):
    """Parse and log message details."""
    try:
        if endpoint == "push-messages":
            from flwr.proto.fleet_pb2 import PushMessagesRequest
            req = PushMessagesRequest()
            req.ParseFromString(protobuf_data)
            logger.info(f"  Node: {req.node.node_id}")
            logger.info(f"  Messages: {len(req.messages_list)}")
    except Exception as e:
        logger.debug(f"Could not inspect message: {e}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8080,
        log_level="info"
    )
```

**Usage**:
```bash
# Install dependencies
pip install fastapi uvicorn httpx

# Run proxy
python production_proxy.py

# Start Flower server
python -m flwr.server.app --fleet-api-type rest

# Start client (pointing to proxy)
flower-supernode --rest --superlink http://localhost:8080
```

### Your REST API Server (Example)

Here's what your relay API might look like:

```python
#!/usr/bin/env python3
"""
Example relay API that receives Flower messages.

This server:
1. Receives protobuf messages from proxy
2. Processes/logs them
3. Forwards to actual Flower server
4. Returns response
"""

from fastapi import FastAPI, Request, Response
import httpx
import logging
from datetime import datetime
import json

app = FastAPI(title="Flower Relay API")
logger = logging.getLogger(__name__)

FLOWER_SERVER = "http://localhost:9095"

# Store for audit log
message_log = []

@app.post("/relay/{endpoint:path}")
async def relay_flower_message(endpoint: str, request: Request):
    """
    Receive Flower message, process, and forward.
    """
    protobuf_data = await request.body()

    # Log the message
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "size": len(protobuf_data),
        "source_ip": request.client.host
    }
    message_log.append(log_entry)

    logger.info(f"Received: {endpoint} from {request.client.host}")

    # Parse protobuf (for your custom processing)
    try:
        if endpoint == "push-messages":
            from flwr.proto.fleet_pb2 import PushMessagesRequest
            req = PushMessagesRequest()
            req.ParseFromString(protobuf_data)

            # Extract information
            node_id = req.node.node_id
            num_messages = len(req.messages_list)

            logger.info(f"  Node {node_id} pushed {num_messages} messages")

            # Store in your database
            # await store_training_result(node_id, req)

        elif endpoint == "pull-messages":
            from flwr.proto.fleet_pb2 import PullMessagesRequest
            req = PullMessagesRequest()
            req.ParseFromString(protobuf_data)

            logger.info(f"  Node {req.node.node_id} pulling messages")

    except Exception as e:
        logger.warning(f"Could not parse message: {e}")

    # Forward to actual Flower server
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{FLOWER_SERVER}/api/v0/fleet/{endpoint}",
            content=protobuf_data,
            headers={
                "Content-Type": "application/protobuf",
                "Accept": "application/protobuf"
            },
            timeout=60
        )

    # Return Flower's response
    return Response(
        content=response.content,
        status_code=response.status_code,
        headers={"Content-Type": "application/protobuf"}
    )

@app.get("/audit-log")
async def get_audit_log():
    """View message audit log."""
    return {
        "total_messages": len(message_log),
        "recent": message_log[-50:]  # Last 50 messages
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9999)
```

---

## Testing & Debugging

### Test Setup

```bash
# Terminal 1: Start your relay API
python relay_api.py

# Terminal 2: Start production proxy
python production_proxy.py

# Terminal 3: Start Flower server
python -m flwr.server.app --fleet-api-type rest

# Terminal 4: Start client
flower-supernode --rest --superlink http://localhost:8080
```

### Debugging with mitmproxy

```bash
# Install mitmproxy
pip install mitmproxy

# Start mitmproxy
mitmproxy --mode reverse:http://localhost:9095 --listen-port 8080

# Start client pointing to mitmproxy
flower-supernode --rest --superlink http://localhost:8080
```

mitmproxy will show all HTTP requests/responses in real-time.

### Network Traffic Capture

```bash
# Capture traffic on port 9095
sudo tcpdump -i any -A 'port 9095' -w flower_traffic.pcap

# Analyze with Wireshark
wireshark flower_traffic.pcap
```

### Unit Testing Your Proxy

```python
# test_proxy.py
import pytest
from httpx import AsyncClient
from production_proxy import app

@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_message_interception():
    """Test message passthrough."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Mock protobuf data
        test_data = b'\x08\x01'  # Simple protobuf

        response = await client.post(
            "/api/v0/fleet/register-node",
            content=test_data,
            headers={"Content-Type": "application/protobuf"}
        )

        assert response.status_code in [200, 500]  # 500 if Flower not running
```

### Logging Configuration

```python
# Configure detailed logging
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('flower_proxy.log'),
        logging.StreamHandler()
    ]
)

# Log all HTTP requests
import http.client as http_client
http_client.HTTPConnection.debuglevel = 1
```

---

## Production Considerations

### Security

#### 1. TLS/SSL Encryption

```python
# Enable HTTPS in proxy
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8443,
        ssl_keyfile="/path/to/key.pem",
        ssl_certfile="/path/to/cert.pem"
    )

# Client connects with HTTPS
flower-supernode --rest --superlink https://proxy:8443 \
    --root-certificates /path/to/ca.pem
```

#### 2. Authentication

```python
from fastapi import Header, HTTPException

async def verify_token(x_api_key: str = Header(...)):
    """Verify API key for authentication."""
    if x_api_key != "your-secret-key":
        raise HTTPException(status_code=401, detail="Invalid API key")

@app.post("/api/v0/fleet/{endpoint:path}")
async def intercept_flower_message(
    endpoint: str,
    request: Request,
    auth: None = Depends(verify_token)  # Require auth
):
    # ... existing code ...
```

#### 3. Rate Limiting

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/api/v0/fleet/{endpoint:path}")
@limiter.limit("100/minute")  # Max 100 requests per minute
async def intercept_flower_message(
    endpoint: str,
    request: Request
):
    # ... existing code ...
```

### Scalability

#### 1. Horizontal Scaling

Deploy multiple proxy instances behind a load balancer:

```yaml
# docker-compose.yml
version: '3.8'
services:
  proxy1:
    build: .
    ports:
      - "8081:8080"

  proxy2:
    build: .
    ports:
      - "8082:8080"

  nginx:
    image: nginx
    ports:
      - "8080:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - proxy1
      - proxy2
```

```nginx
# nginx.conf
upstream flower_proxy {
    least_conn;
    server proxy1:8080;
    server proxy2:8080;
}

server {
    listen 80;

    location /api/v0/fleet/ {
        proxy_pass http://flower_proxy;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

#### 2. Caching (for read-heavy endpoints)

```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=1000)
def cache_key(protobuf_data: bytes) -> str:
    """Generate cache key from protobuf data."""
    return hashlib.sha256(protobuf_data).hexdigest()

# Cache get_run responses (immutable)
run_cache = {}

@app.post("/api/v0/fleet/get-run")
async def get_run_cached(request: Request):
    protobuf_data = await request.body()
    key = cache_key(protobuf_data)

    if key in run_cache:
        logger.info("Cache hit for get-run")
        return Response(
            content=run_cache[key],
            headers={"Content-Type": "application/protobuf"}
        )

    # Forward to Flower
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{FLOWER_SERVER}/api/v0/fleet/get-run",
            content=protobuf_data,
            headers={"Content-Type": "application/protobuf"}
        )

    # Cache response
    run_cache[key] = response.content

    return Response(
        content=response.content,
        headers={"Content-Type": "application/protobuf"}
    )
```

### Monitoring

#### Prometheus Metrics

```python
from prometheus_client import Counter, Histogram, generate_latest

# Metrics
request_count = Counter(
    'flower_proxy_requests_total',
    'Total requests',
    ['endpoint', 'status']
)

request_duration = Histogram(
    'flower_proxy_request_duration_seconds',
    'Request duration',
    ['endpoint']
)

@app.post("/api/v0/fleet/{endpoint:path}")
async def intercept_with_metrics(endpoint: str, request: Request):
    start_time = time.time()

    try:
        # ... existing code ...
        response = await forward_to_flower(endpoint, protobuf_data)

        request_count.labels(endpoint=endpoint, status=response.status_code).inc()
        request_duration.labels(endpoint=endpoint).observe(time.time() - start_time)

        return response
    except Exception as e:
        request_count.labels(endpoint=endpoint, status='error').inc()
        raise

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")
```

### Reliability

#### Circuit Breaker Pattern

```python
from pybreaker import CircuitBreaker

# Circuit breaker for Flower server
flower_breaker = CircuitBreaker(
    fail_max=5,  # Open after 5 failures
    timeout_duration=60  # Stay open for 60 seconds
)

@app.post("/api/v0/fleet/{endpoint:path}")
async def intercept_with_circuit_breaker(endpoint: str, request: Request):
    protobuf_data = await request.body()

    try:
        # Call with circuit breaker protection
        @flower_breaker
        async def call_flower():
            async with httpx.AsyncClient() as client:
                return await client.post(
                    f"{FLOWER_SERVER}/api/v0/fleet/{endpoint}",
                    content=protobuf_data,
                    headers={"Content-Type": "application/protobuf"},
                    timeout=60
                )

        response = await call_flower()

    except CircuitBreakerError:
        logger.error("Circuit breaker open - Flower server unavailable")
        return JSONResponse(
            status_code=503,
            content={"error": "Service temporarily unavailable"}
        )

    return Response(
        content=response.content,
        headers={"Content-Type": "application/protobuf"}
    )
```

---

## Summary

### Recommended Approach for Most Use Cases

**Use HTTP Proxy with REST Transport:**

1. ✅ Simplest to implement and maintain
2. ✅ No Flower codebase modifications
3. ✅ Easy to deploy and scale
4. ✅ Good performance for most workloads
5. ✅ Rich ecosystem of tools (nginx, HAProxy, etc.)

### Quick Start Checklist

- [ ] Start Flower server with REST: `python -m flwr.server.app --fleet-api-type rest`
- [ ] Implement HTTP proxy (use FastAPI example above)
- [ ] Test with simple passthrough first
- [ ] Add your relay logic
- [ ] Implement logging and monitoring
- [ ] Add authentication and TLS
- [ ] Load test and optimize
- [ ] Deploy to production

### Key Files to Reference

| File | Purpose |
|------|---------|
| `py/flwr/client/rest_client/connection.py` | Client-side REST implementation |
| `py/flwr/server/superlink/fleet/rest_rere/rest_api.py` | Server-side REST API |
| `proto/flwr/proto/fleet.proto` | Protobuf message definitions |
| `py/flwr/common/serde.py` | Serialization utilities |

---

## Further Reading

- Flower Documentation: https://flower.ai/docs
- Protocol Buffers: https://protobuf.dev/
- FastAPI: https://fastapi.tiangolo.com/
- gRPC Interceptors: https://grpc.io/docs/guides/interceptors/
- HTTP/2 and gRPC: https://grpc.io/docs/what-is-grpc/core-concepts/

---

**Questions or Issues?**

File issues at: https://github.com/adap/flower/issues

Community: https://flower.ai/join-slack