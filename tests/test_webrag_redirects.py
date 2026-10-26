"""Offline regression tests for redirect handling in webrag.net: credential headers never cross origins, every hop
is re-validated, malformed or endless redirects fail as FetchError, and no credential value reaches an error or a
stored URL (no network)."""
import pytest

from webrag.net import FetchError, PoliteSession, origin_of, redact

SECRET = "S3CRET-VALUE"
CREDENTIALS = {"Authorization": f"Bearer {SECRET}", "proxy-authorization": f"Basic {SECRET}",
               "X-SUBSCRIPTION-TOKEN": SECRET, "Cookie": f"session={SECRET}", "X-Api-Key": SECRET}


class FakeResponse:
    def __init__(self, status=200, body=b"ok", headers=None, url="https://api.example.org/a"):
        self.status_code, self.body, self.url = status, body, url
        self.headers = headers if headers is not None else {"Content-Type": "text/html"}

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass


def redirect(location, url):
    return FakeResponse(302, b"", {"Location": location}, url)


def session_with(responses, resolver=lambda host: ["93.184.216.34"], **kwargs):
    kwargs.setdefault("respect_robots", False)
    session = PoliteSession(min_interval=0, resolver=resolver, **kwargs)
    queue, session.calls = list(responses), []

    def fake_get(url, **kw):
        session.calls.append({"url": url, **kw})
        return queue.pop(0)

    session.session.get = fake_get
    return session


def sent_headers(call):
    return call.get("headers") or {}


# ------------------------------------------------------------------------------------------------ origin definition

@pytest.mark.parametrize("a,b,same", [
    ("https://api.example.org/a", "https://API.example.org:443/b?x=1", True),       # case and default port
    ("http://api.example.org/", "http://api.example.org:80/", True),
    ("https://api.example.org/", "http://api.example.org/", False),                 # scheme
    ("https://api.example.org/", "https://api.example.org:8443/", False),           # port
    ("https://api.example.org/", "https://example.org/", False),                    # host (parent domain)
    ("https://user:pw@api.example.org/", "https://api.example.org/", True),         # user-info is not part of it
    ("http://[::1]:8080/", "http://[::1]:8080/x", True),
])
def test_origin_is_scheme_host_and_effective_port(a, b, same):
    assert (origin_of(a) == origin_of(b)) is same


# ------------------------------------------------------------------------------------------------ credential headers

def test_same_origin_redirects_keep_headers():
    session = session_with([redirect("/b", "https://api.example.org/a"),
                            redirect("https://api.example.org:443/c", "https://api.example.org/b"),
                            FakeResponse(url="https://api.example.org/c")])
    headers = dict(CREDENTIALS, Accept="application/json")
    page = session.get("https://api.example.org/a", headers=headers)
    assert page.content == b"ok" and len(session.calls) == 3
    assert all(sent_headers(c) == headers for c in session.calls)


@pytest.mark.parametrize("location", ["https://other.example.net/x",       # another host
                                      "//other.example.net/x",              # scheme-relative, another host
                                      "http://api.example.org/x",           # https -> http downgrade
                                      "https://api.example.org:8443/x",     # another port
                                      "https://example.org/x"])             # parent domain
def test_cross_origin_redirects_drop_credential_headers(location):
    session = session_with([redirect(location, "https://api.example.org/a"), FakeResponse(url=location)])
    session.get("https://api.example.org/a", headers=dict(CREDENTIALS, Accept="application/json"),
                params={"api_key": SECRET, "q": "x"})
    first, second = session.calls
    assert sent_headers(first) == dict(CREDENTIALS, Accept="application/json")
    assert sent_headers(second) == {"Accept": "application/json"}           # non-credential headers survive
    assert second["params"] is None and SECRET not in second["url"]          # query parameters are not re-sent


def test_credentials_stay_dropped_when_the_chain_returns_to_the_first_origin():
    session = session_with([redirect("https://other.example.net/b", "https://api.example.org/a"),
                            redirect("https://api.example.org/c", "https://other.example.net/b"),
                            FakeResponse(url="https://api.example.org/c")])
    session.get("https://api.example.org/a", headers=dict(CREDENTIALS))
    assert sent_headers(session.calls[0]) == CREDENTIALS
    assert sent_headers(session.calls[1]) == {} and sent_headers(session.calls[2]) == {}


def test_robots_txt_requests_never_carry_credentials():
    allow = FakeResponse(body=b"User-agent: *\nAllow: /\n", headers={"Content-Type": "text/plain"})
    session = session_with([allow, redirect("https://other.example.net/b", "https://api.example.org/a"),
                            allow, FakeResponse(url="https://other.example.net/b")], respect_robots=True)
    session.get("https://api.example.org/a", headers=dict(CREDENTIALS))
    urls = [c["url"] for c in session.calls]
    assert urls == ["https://api.example.org/robots.txt", "https://api.example.org/a",
                    "https://other.example.net/robots.txt", "https://other.example.net/b"]
    assert sent_headers(session.calls[0]) == {} and sent_headers(session.calls[2]) == {}
    assert sent_headers(session.calls[1]) == CREDENTIALS and sent_headers(session.calls[3]) == {}


# ------------------------------------------------------------------------------------------------ hop validation

@pytest.mark.parametrize("location,code", [
    ("http://127.0.0.1/admin", "blocked_host"),
    ("http://169.254.169.254/latest/meta-data/", "blocked_host"),            # cloud metadata address
    ("http://10.1.2.3/", "blocked_host"),
    ("http://0.0.0.0/", "blocked_host"),
    ("http://[::1]/", "blocked_host"),
    ("http://[::ffff:127.0.0.1]/", "blocked_host"),                         # IPv4-mapped IPv6 loopback
    ("http://intranet.localhost/", "blocked_host"),
    ("file:///etc/passwd", "invalid_url"),
    ("ftp://public.org/x", "invalid_url"),
])
def test_redirects_to_private_or_disallowed_destinations_remain_blocked(location, code):
    session = session_with([redirect(location, "https://public.org/a")])
    with pytest.raises(FetchError) as info:
        session.get("https://public.org/a", headers=dict(CREDENTIALS))
    assert info.value.code == code and len(session.calls) == 1               # the destination is never contacted
    assert SECRET not in str(info.value)


def test_redirect_to_a_name_resolving_to_a_private_address_is_blocked():
    resolver = lambda host: ["10.0.0.7"] if host == "evil.example.net" else ["93.184.216.34"]
    session = session_with([redirect("https://evil.example.net/x", "https://public.org/a")], resolver=resolver)
    with pytest.raises(FetchError, match="resolves to a private") as info:
        session.get("https://public.org/a")
    assert info.value.code == "blocked_host" and len(session.calls) == 1


def test_redirect_from_another_origin_into_the_configured_local_origin_is_blocked():
    """The local-origin exemption (a self-hosted SearXNG) covers requests that start there, not redirects into it."""
    local = "http://127.0.0.1:8888"
    session = session_with([FakeResponse(url=f"{local}/search")], allowed_origins=[f"{local}/"])
    assert session.get(f"{local}/search").status == 200                     # direct use still works
    bounced = session_with([redirect(f"{local}/search", "https://public.org/a")], allowed_origins=[local])
    with pytest.raises(FetchError) as info:
        bounced.get("https://public.org/a")
    assert info.value.code == "blocked_host" and len(bounced.calls) == 1
    internal = session_with([redirect("/search?q=2", f"{local}/search"), FakeResponse(url=f"{local}/search?q=2")],
                            allowed_origins=["HTTP://127.0.0.1:8888"])
    assert internal.get(f"{local}/search").status == 200                    # same-origin hop inside it is fine


@pytest.mark.parametrize("location", ["http://[::1", "https://public.org:99999/x", "https://public.org:abc/x",
                                      "http://", "http://:80/x"])
def test_malformed_redirect_locations_fail_safely(location):
    session = session_with([redirect(location, "https://public.org/a")])
    with pytest.raises(FetchError) as info:
        session.get("https://public.org/a")
    assert info.value.code == "invalid_url" and len(session.calls) == 1


def test_redirect_without_a_location_fails_safely():
    session = session_with([FakeResponse(302, b"", {}, "https://public.org/a")])
    with pytest.raises(FetchError, match="without a location"):
        session.get("https://public.org/a")


def test_redirect_chains_are_limited():
    hops = [redirect(f"/r{i + 1}", f"https://public.org/r{i}") for i in range(3)]
    session = session_with(hops, max_redirects=2)
    with pytest.raises(FetchError, match="too many redirects") as info:
        session.get(f"https://public.org/r0?api_key={SECRET}")
    assert len(session.calls) == 3 and SECRET not in str(info.value)
    ok = session_with(hops[:2] + [FakeResponse(url="https://public.org/r2")], max_redirects=2)
    assert ok.get("https://public.org/r0").status == 200                    # exactly at the limit still works


# ------------------------------------------------------------------------------------------------ no credential values

def test_credential_values_never_reach_errors_or_stored_urls():
    assert redact(f"https://user:{SECRET}@a.org/x?token={SECRET}&q=1") == \
        "https://REDACTED@a.org/x?token=REDACTED&q=1"
    url = f"https://user:{SECRET}@public.org/x?api_key={SECRET}&access_token={SECRET}"
    failing = session_with([FakeResponse(404, url=url)])
    with pytest.raises(FetchError) as info:
        failing.get(url, headers=dict(CREDENTIALS))
    assert SECRET not in str(info.value)
    stored = session_with([FakeResponse(url=url)]).get(url, headers=dict(CREDENTIALS))
    assert SECRET not in stored.url and "REDACTED" in stored.url
    malformed = session_with([redirect(f"http://[::1?api_key={SECRET}", url)])
    with pytest.raises(FetchError) as info:
        malformed.get(url)
    assert SECRET not in str(info.value)
