"""Offline tests for the address policy of webrag.net (SSRF guard): which destinations are allowed, NAT64 (RFC 6052
well-known prefix) and CGNAT handling, mixed DNS answers, redirects, numeric host encodings, proxies, TLS settings and
error codes. Resolvers and the HTTP layer are fakes; no socket is opened and nothing leaves this computer."""
import ipaddress

import pytest
import requests

import webrag.net as net
from webrag.net import FetchError, PoliteSession, _blocked_address, nat64_embedded


class FakeResponse:
    def __init__(self, status=200, body=b"ok", headers=None, url="https://public.example.org/a"):
        self.status_code, self.body, self.url = status, body, url
        self.headers = headers if headers is not None else {"Content-Type": "text/html"}

    def iter_content(self, size):
        yield self.body

    def close(self):
        pass


def session_with(responses=(), resolver=lambda host: ["93.184.216.34"], **kwargs):
    """A PoliteSession whose transport is a queue of responses (or exceptions); every call is recorded."""
    kwargs.setdefault("respect_robots", False)
    session = PoliteSession(min_interval=0, resolver=resolver, **kwargs)
    queue, session.calls = list(responses), []

    def fake_get(url, **kw):
        session.calls.append({"url": url, **kw})
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    session.session.get = fake_get
    return session


@pytest.fixture(autouse=True)
def no_proxies(monkeypatch):
    """Tests run without proxy settings unless a test sets them (a proxy in the developer's environment would
    otherwise trigger the proxy policy)."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "WEBRAG_ALLOW_PROXY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(requests.utils, "get_environ_proxies", lambda url, no_proxy=None: {})


# ------------------------------------------------------------------------------------------------ address policy
@pytest.mark.parametrize("address", ["93.184.216.34", "151.101.159.42", "2606:4700:83b4:dec5:4bfd:aa0:c3a9:3c5f",
                                     "2001:4860:4860::8888"])
def test_public_addresses_are_allowed(address):
    assert not _blocked_address(ipaddress.ip_address(address))


@pytest.mark.parametrize("address,what", [
    ("127.0.0.1", "loopback"), ("::1", "loopback v6"), ("10.0.0.5", "private"), ("172.16.0.1", "private"),
    ("192.168.1.1", "private"), ("169.254.169.254", "cloud metadata / link-local"), ("fe80::1", "link-local v6"),
    ("0.0.0.0", "unspecified"), ("::", "unspecified v6"), ("224.0.0.1", "multicast"), ("ff02::1", "multicast v6"),
    ("240.0.0.1", "reserved"), ("100.64.0.1", "CGNAT"), ("100.127.255.254", "CGNAT"), ("fd00::1", "unique local"),
    ("192.0.2.1", "documentation"), ("2001:db8::1", "documentation v6"), ("198.18.0.1", "benchmarking"),
    ("::ffff:127.0.0.1", "IPv4-mapped loopback"), ("2002:7f00:1::1", "6to4 embedding loopback"),
])
def test_prohibited_addresses_are_refused(address, what):
    assert _blocked_address(ipaddress.ip_address(address)), what


def test_nat64_well_known_prefix_is_judged_by_the_embedded_ipv4():
    assert nat64_embedded(ipaddress.ip_address("64:ff9b::9765:9f2a")) == ipaddress.IPv4Address("151.101.159.42")
    assert not _blocked_address(ipaddress.ip_address("64:ff9b::9765:9f2a"))         # what DNS64 returned for arXiv
    for embedded in ("7f00:1", "a00:1", "a9fe:a9fe", "6440:1", "0:0", "c0a8:101", "e000:1"):
        # 127.0.0.1, 10.0.0.1, 169.254.169.254, 100.64.0.1, 0.0.0.0, 192.168.1.1, 224.0.0.1
        assert _blocked_address(ipaddress.ip_address(f"64:ff9b::{embedded}")), embedded


def test_other_translation_prefixes_are_not_decoded():
    assert nat64_embedded(ipaddress.ip_address("64:ff9b:1::9765:9f2a")) is None     # RFC 8215 local-use
    assert _blocked_address(ipaddress.ip_address("64:ff9b:1::9765:9f2a"))
    # IPv4-mapped addresses are judged by their IPv4 address, unchanged from before: public allowed, loopback refused
    assert not _blocked_address(ipaddress.ip_address("::ffff:151.101.159.42"))
    assert _blocked_address(ipaddress.ip_address("::ffff:127.0.0.1"))
    assert nat64_embedded(ipaddress.ip_address("2606:4700::9765:9f2a")) is None       # an ordinary global address


# ------------------------------------------------------------------------------------------------ DNS answers
@pytest.mark.parametrize("answer,allowed", [
    (["151.101.159.42", "64:ff9b::9765:9f2a"], True),        # the DNS64 answer observed on a live network
    (["151.101.159.42"], True),
    (["151.101.159.42", "10.0.0.5"], False),                  # one prohibited address refuses the host
    (["151.101.159.42", "64:ff9b::7f00:1"], False),           # a NAT64 address hiding loopback
    (["151.101.159.42", "64:ff9b:1::9765:9f2a"], False),      # an undecoded prefix
    (["100.64.0.1"], False),
    ([], False),                                              # no answer is not an allowance
])
def test_mixed_dns_answers_cannot_bypass_the_policy(answer, allowed):
    session = session_with([FakeResponse()], resolver=lambda host: answer)
    if allowed:
        assert session.get("https://export.arxiv.example/api").status == 200
    else:
        with pytest.raises(FetchError) as exc:
            session.get("https://export.arxiv.example/api")
        assert exc.value.code == "blocked_host" and session.calls == []          # nothing was sent


def test_a_cached_decision_is_reused_for_the_host():
    answers = iter([["151.101.159.42"], ["127.0.0.1"]])
    session = session_with([FakeResponse(), FakeResponse()], resolver=lambda host: next(answers))
    session.get("https://cached.example.org/a")
    session.get("https://cached.example.org/b")          # a second, different answer is not consulted
    assert len(session.calls) == 2


def test_connection_is_bound_to_the_checked_addresses():
    """The transport connects to the checked addresses (PinnedAdapter); the real handshakes are in
    tests/test_webrag_tls_pinning.py."""
    session = PoliteSession()
    assert type(session.session.get_adapter("https://example.org/")).__name__ == "PinnedAdapter"
    assert type(session.session.get_adapter("http://example.org/")).__name__ == "PinnedAdapter"


def test_the_checked_addresses_are_bound_to_the_request_ipv4_first_and_rechecked_after_the_ttl(monkeypatch):
    answers = iter([["2606:4700::6810:84e5", "151.101.159.42"], ["127.0.0.1"]])
    session = session_with([FakeResponse(), FakeResponse()], resolver=lambda host: next(answers))
    pins = []
    real_get = session.session.get
    session.session.get = lambda url, **kw: pins.append(session._pin.value) or real_get(url, **kw)
    session.get("https://ttl.example.org/a")
    assert pins == [("ttl.example.org", ("151.101.159.42", "2606:4700::6810:84e5"))]
    assert session._pin.value is None                  # cleared after the request
    monkeypatch.setattr(net, "ADDRESS_TTL", -1.0)
    with pytest.raises(FetchError) as exc:
        session.get("https://ttl.example.org/b")
    assert exc.value.code == "blocked_host" and len(session.calls) == 1


# ------------------------------------------------------------------------------------------------ URLs and redirects
@pytest.mark.parametrize("url", ["http://2130706433/", "http://0x7f.1/", "http://127.1/", "http://017700000001/",
                                 "http://0x7f000001/", "http://localhost./", "http://LOCALHOST/", "http://printer.local/",
                                 "http://[::1]/", "http://[64:ff9b::7f00:1]/", "http://100.64.0.1/"])
def test_numeric_encodings_local_names_and_literals_are_refused_before_any_lookup(url):
    looked_up = []
    session = session_with([FakeResponse()], resolver=lambda host: looked_up.append(host) or ["93.184.216.34"])
    with pytest.raises(FetchError) as exc:
        session.get(url)
    assert exc.value.code == "blocked_host" and looked_up == [] and session.calls == []


@pytest.mark.parametrize("url", ["http://[64:ff9b::9765:9f2a]/x", "https://93.184.216.34/x", "https://public.example.org:8443/x"])
def test_public_literals_and_unusual_ports_are_allowed(url):
    assert session_with([FakeResponse(url=url)]).get(url).status == 200


@pytest.mark.parametrize("url", ["ftp://public.example.org/", "https://public.example.org:99999/", "http:///nohost", "javascript:alert(1)"])
def test_malformed_urls_are_invalid(url):
    with pytest.raises(FetchError) as exc:
        session_with().get(url)
    assert exc.value.code == "invalid_url"


@pytest.mark.parametrize("location", ["http://[64:ff9b::a9fe:a9fe]/latest/meta-data", "http://0x7f.1/admin",
                                      "http://100.64.0.10/", "https://rebind.example.net/x"])
def test_every_redirect_hop_is_checked(location):
    resolver = lambda host: ["64:ff9b::7f00:1"] if host == "rebind.example.net" else ["93.184.216.34"]
    session = session_with([FakeResponse(302, b"", {"Location": location}, "https://public.example.org/a"), FakeResponse()],
                           resolver=resolver)
    with pytest.raises(FetchError) as exc:
        session.get("https://public.example.org/a")
    assert exc.value.code == "blocked_host" and len(session.calls) == 1        # the second hop was never sent


# ------------------------------------------------------------------------------------------------ TLS and proxies
def test_tls_verification_is_left_on():
    session = session_with([FakeResponse()])
    assert session.session.verify is True
    session.get("https://public.example.org/a")
    assert session.calls[0].get("verify", True) is True                          # never turned off per request


def test_an_environment_proxy_is_refused_unless_accepted(monkeypatch):
    monkeypatch.setattr(requests.utils, "get_environ_proxies", lambda url, no_proxy=None: {"https": "http://10.0.0.5:3128"})
    session = session_with([FakeResponse()])
    with pytest.raises(FetchError) as exc:
        session.get("https://public.example.org/a")
    assert exc.value.code == "proxy_policy" and "WEBRAG_ALLOW_PROXY=1" in str(exc.value) and session.calls == []
    monkeypatch.setenv("WEBRAG_ALLOW_PROXY", "1")                                  # an explicit, documented opt-in
    assert session_with([FakeResponse()]).get("https://public.example.org/a").status == 200


def test_proxy_lookup_errors_refuse_rather_than_guess(monkeypatch):
    def broken(url, no_proxy=None):
        raise ValueError("bad proxy setting")
    monkeypatch.setattr(requests.utils, "get_environ_proxies", broken)
    with pytest.raises(FetchError) as exc:
        session_with([FakeResponse()]).get("https://public.example.org/a")
    assert exc.value.code == "proxy_policy"


# ------------------------------------------------------------------------------------------------ failure codes
@pytest.mark.parametrize("error,code", [
    (requests.exceptions.SSLError("certificate verify failed"), "tls"),
    (requests.exceptions.ConnectionError("NameResolutionError: failed to resolve"), "dns"),
    (requests.exceptions.ConnectionError("Connection refused"), "network"),
    (requests.exceptions.Timeout("read timed out"), "timeout"),
    (requests.exceptions.ChunkedEncodingError("broken"), "network"),
])
def test_transport_failures_are_distinguished(error, code):
    with pytest.raises(FetchError) as exc:
        session_with([error]).get("https://public.example.org/a?api_key=S3CRET")
    assert exc.value.code == code and "S3CRET" not in str(exc.value)


def test_proxy_failure_has_its_own_code(monkeypatch):
    monkeypatch.setenv("WEBRAG_ALLOW_PROXY", "1")
    with pytest.raises(FetchError) as exc:
        session_with([requests.exceptions.ProxyError("proxy down")]).get("https://public.example.org/a")
    assert exc.value.code == "proxy"


def test_dns_failure_and_policy_refusal_are_different_codes():
    def fails(host):
        raise OSError("no such host")
    with pytest.raises(FetchError) as exc:
        session_with(resolver=fails).get("https://missing.example.org/a")
    assert exc.value.code == "dns"
    with pytest.raises(FetchError) as exc:
        session_with(resolver=lambda host: ["10.0.0.1"]).get("https://internal.example.org/a")
    assert exc.value.code == "blocked_host"


def test_retries_are_bounded_and_never_follow_a_refusal():
    limited = FakeResponse(429, b"", {"Retry-After": "0"})
    session = session_with([limited, FakeResponse(429, b"", {"Retry-After": "0"})])
    with pytest.raises(FetchError) as exc:
        session.get("https://public.example.org/a")
    assert exc.value.code == "rate_limited" and len(session.calls) == 2          # one retry, then stop
    refused = session_with([FakeResponse()], resolver=lambda host: ["127.0.0.1"])
    for _ in range(3):
        with pytest.raises(FetchError):
            refused.get("https://refused.example.org/a")
    assert refused.calls == []
