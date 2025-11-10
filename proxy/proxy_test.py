#! /usr/bin/env python3

from flask import Flask, request, Response
import requests
from argparse import ArgumentParser
import logging

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

app = Flask(__name__)

def create_app():
    app.config.from_prefixed_env()
    return app


# Configuration
FLOWER_SERVER = "http://localhost:9095"  # Actual Flower SuperLink
YOUR_API = "https://your-api.com"        # Your REST API

@app.route('/api/v0/fleet/<path:endpoint>', methods=['POST'])
def proxy_flower_messages(endpoint):
    """Intercept all Flower messages."""

    flower_server = app.config["SUPERLINK"]
    logger.info(f"Connecting to superlink at {flower_server}")

    # 1. Receive protobuf from SuperNode
    protobuf_data = request.data
    headers = {
        'Content-Type': 'application/protobuf',
        'Accept': 'application/protobuf'
    }

    logging.info(f"[INTERCEPT] Endpoint: {endpoint}")
    logging.info(f"[INTERCEPT] Data size: {len(protobuf_data)} bytes")

    # 2. Forward to Flower server
    flower_response = requests.post(
        f"{flower_server}/api/v0/fleet/{endpoint}",
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

def main():
    parser = ArgumentParser()

    parser.add_argument("--superlink", type=str, default=FLOWER_SERVER)
    parser.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    app.config['SUPERLINK'] = args.superlink
    app.run(host='0.0.0.0', port=args.port)

if __name__ == '__main__':
    main()
