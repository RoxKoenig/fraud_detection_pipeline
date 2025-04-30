from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
import httpx
import logging

app = FastAPI()

# Setup basic logging
logging.basicConfig(level=logging.INFO)

# Internal MLflow backend service inside Docker network
MLFLOW_BACKEND = "http://mlflow_server:5001"

# Map tokens to user roles (if you want Bearer token auth)
TOKEN_ROLE_MAP = {
    "viewer-token": "viewer",
    "writer-token": "writer",
    "admin-token": "admin",
}

# Define what each role can do
ROLE_PERMISSIONS = {
    "viewer": {"GET"},
    "writer": {"GET", "POST", "PUT"},
    "admin": {"GET", "POST", "PUT", "DELETE"},
}

# Restrict DELETE actions to admins only
RESTRICTED_PATHS = {
    "DELETE": [
        "/api/2.0/mlflow/experiments/delete",
        "/api/2.0/mlflow/runs/delete",
    ]
}

# Helper function to get role
def get_role(token: str):
    return TOKEN_ROLE_MAP.get(token)

# Helper function to check if role is allowed
def is_allowed(role: str, method: str, path: str):
    if method.upper() == "OPTIONS":
        return True

    if method.upper() not in ROLE_PERMISSIONS.get(role, set()):
        return False

    for restricted_path in RESTRICTED_PATHS.get(method.upper(), []):
        if path.startswith(restricted_path) and role != "admin":
            return False

    return True

# Main middleware for RBAC and proxying
@app.middleware("http")
async def rbac_token_middleware(request: Request, call_next):
    path = request.url.path
    method = request.method

    auth = request.headers.get("Authorization")
    role = None

    # Token-based auth
    if auth and auth.startswith("Bearer "):
        token = auth.split(" ")[1]
        role = get_role(token)
    else:
        # Fallback to NGINX basic auth role header
        role = request.headers.get("X-User-Role")

    logging.info(f"[{method}] {path} requested by role='{role}'")

    if not role:
        return JSONResponse({"detail": "Unauthorized or missing role"}, status_code=403)

    if not is_allowed(role, method, path):
        return JSONResponse(
            {"detail": f"'{role}' not allowed to perform '{method}' on {path}"},
            status_code=403,
        )

    # Proxy request to MLflow backend
    try:
        async with httpx.AsyncClient() as client:
            mlflow_url = f"{MLFLOW_BACKEND}{path}"

            forwarded_headers = {
                k: v for k, v in request.headers.items()
                if k.lower() not in {"content-length", "content-encoding"}
            }

            # Correct the Host header manually
            forwarded_headers["host"] = MLFLOW_BACKEND.replace("http://", "").split("/")[0]

            resp = await client.request(
                method=method,
                url=mlflow_url,
                headers=forwarded_headers,
                params=request.query_params,
                content=await request.body(),
                timeout=60.0,
            )

            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers={k: v for k, v in resp.headers.items() if k.lower() not in {"content-length", "transfer-encoding"}},
                media_type=resp.headers.get("content-type", "application/octet-stream"),
            )

    except httpx.RequestError as e:
        return JSONResponse(
            {"detail": f"Error contacting MLflow backend: {str(e)}"},
            status_code=502,
        )
