"""The API's documentation: the OpenAPI schema and the three pages that render it (Swagger UI, ReDoc, Scalar)."""

from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest
from fastapi.testclient import TestClient

from my_diffusion import api


@pytest.fixture(scope="module")
def client():
    return TestClient(api.app)   # no lifespan needed: the documentation does not touch the models


@pytest.fixture(scope="module")
def schema(client):
    return client.get("/openapi.json").json()


def test_schema_lists_the_job_flow(schema):
    for path in ("/generate", "/result/{job_id}", "/result/{job_id}/image", "/cancel/{job_id}", "/jobs"):
        assert path in schema["paths"]
    assert schema["info"]["title"] == "my-diffusion"
    assert {tag["name"] for tag in schema["tags"]} == {"jobs", "models", "service", "config"}


def test_the_key_and_the_user_id_are_security_schemes(schema):
    # As schemes (not plain headers) they get an "Authorize" button in Swagger UI and Scalar.
    schemes = schema["components"]["securitySchemes"]
    assert schemes["ApiKey"] == {**schemes["ApiKey"], "type": "http", "scheme": "bearer"}
    assert schemes["UserId"]["type"] == "apiKey" and schemes["UserId"]["in"] == "header"
    assert schemes["UserId"]["name"] == "X-User-Id"


def test_protected_endpoints_declare_the_key_and_open_ones_do_not(schema):
    generate = schema["paths"]["/generate"]["post"]
    assert {"ApiKey": []} in generate["security"]
    assert "security" not in schema["paths"]["/health"]["get"]
    assert "security" not in schema["paths"]["/resolutions"]["get"]


def test_error_answers_are_documented(schema):
    generate = schema["paths"]["/generate"]["post"]["responses"]
    assert {"202", "400", "401", "429"} <= set(generate)
    assert "200" not in generate                      # it answers 202, not 200
    image = schema["paths"]["/result/{job_id}/image"]["get"]["responses"]
    assert "image/png" in image["200"]["content"]
    assert "409" in image


def test_the_request_has_a_working_example(schema):
    example = schema["components"]["schemas"]["GenerateRequest"]["examples"][0]
    assert example["prompt"] and example["resolution"] == "720"


@pytest.mark.parametrize("path, marker", [("/docs", "swagger-ui"), ("/redoc", "redoc"), ("/scalar", "createApiReference")])
def test_documentation_pages_are_served(client, path, marker):
    page = client.get(path)
    assert page.status_code == 200 and "text/html" in page.headers["content-type"]
    assert marker in page.text


def test_scalar_is_pinned_and_integrity_checked(client):
    page = client.get("/scalar").text
    assert f"@scalar/api-reference@{api.SCALAR_VERSION}/" in page
    assert re.fullmatch(r"\d+\.\d+\.\d+", api.SCALAR_VERSION)          # an exact version, never "latest"
    assert f'integrity="{api.SCALAR_SRI}"' in page and 'crossorigin="anonymous"' in page
    assert re.fullmatch(r"sha(256|384|512)-[A-Za-z0-9+/]+=*", api.SCALAR_SRI)
    assert "url: '/openapi.json'" in page


def test_scalar_cloud_features_are_off(client):
    # "Ask AI", the SDK / MCP generators and telemetry talk to Scalar's servers: a local API has no use for them.
    page = client.get("/scalar").text
    for setting in ("showDeveloperTools: 'never'", "agent: { disabled: true }", "mcp: { disabled: true }",
                    "telemetry: false", "persistAuth: false"):
        assert setting in page


def test_documentation_can_be_turned_off():
    assert api.docs_urls(False) == {"docs_url": None, "redoc_url": None, "openapi_url": None}
    assert api.docs_urls(True)["openapi_url"] == "/openapi.json"


def test_turned_off_documentation_leaves_no_page(tmp_path):
    # The setting is read at import, so a fresh interpreter checks the real thing: no route left at all.
    code = (
        "from my_diffusion.api import app; "
        "print(sorted(r.path for r in app.routes if r.path in ('/docs', '/redoc', '/scalar', '/openapi.json')))"
    )
    env = {**os.environ, "MY_DIFFUSION_API_DOCS": "false", "MY_DIFFUSION_OUTPUT_DIR": str(tmp_path)}
    done = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "[]"
