#!/usr/bin/env python3
"""
Travion AI Engine Runner Script.

This script starts the FastAPI application with uvicorn.

Usage:
    python run.py                    # Development mode (auto-reload)
    python run.py --production       # Production mode

Environment Variables:
    PORT: Server port (default: 8000)
    HOST: Server host (default: 0.0.0.0)
    DEBUG: Enable debug mode (default: true)
"""

import argparse
import os
import sys

import uvicorn


def main():
    """Run the Travion AI Engine server."""
    parser = argparse.ArgumentParser(description="Travion AI Engine Server")
    parser.add_argument(
        "--production",
        action="store_true",
        help="Run in production mode (no auto-reload)"
    )
    parser.add_argument(
        "--host",
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("PORT", 8000)),
        help="Port to bind to (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (production only)"
    )
    args = parser.parse_args()

    # Configure uvicorn
    config = {
        "app": "app.main:app",
        "host": args.host,
        "port": args.port,
        "log_level": "info",
    }

    if args.production:
        # Production settings
        config["workers"] = args.workers
        config["reload"] = False
        print(f"Starting Travion AI Engine in PRODUCTION mode...")
        print(f"Workers: {args.workers}")
    else:
        # Development settings
        config["reload"] = True
        config["reload_dirs"] = ["app"]
        print(f"Starting Travion AI Engine in DEVELOPMENT mode...")
        print(f"Auto-reload enabled")

    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print(f"Health Check: http://{args.host}:{args.port}/api/v1/health")
    print("-" * 50)

    # Run the server
    uvicorn.run(**config)


if __name__ == "__main__":
    main()
