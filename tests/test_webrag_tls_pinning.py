"""Offline tests for connection pinning in webrag.net (PinnedAdapter): the connection goes to the address the guard
checked, never to a fresh DNS answer, while TLS still verifies the certificate chain and the original host name and
the original name is sent as SNI and Host.

Everything runs on loopback. A throwaway CA and server certificates are made with the system ``openssl`` command in
pytest's temporary folder (nothing is added to any certificate store); the tests are skipped when ``openssl`` is not
installed. The guard's resolver is a fake, ``socket.getaddrinfo`` refuses every ``*.test`` name (so any attempt to
resolve the name again fails the request and is recorded), and the address policy is relaxed for 127.0.0.0/8 only,
since the test servers can only live there. No request leaves this computer."""
import http.server
import ipaddress
import os
import shutil
import socket
import ssl
import subprocess
import threading

import pytest
import requests

import webrag.net as net
from webrag.net import FetchError, PinnedAdapter, PinningError, PoliteSession

GIT_OPENSSL = [r"C:\Program Files\Git\usr\bin\openssl.exe", r"C:\Program Files\Git\mingw64\bin\openssl.exe"]
OPENSSL = shutil.which("openssl") or next((p for p in GIT_OPENSSL if os.path.exists(p)), None)
pytestmark = pytest.mark.skipif(OPENSSL is None, reason="the openssl command is not installed; the loopback TLS tests "
                                                         "need it to make throwaway certificates")


# ------------------------------------------------------------------------------------------------ certificates
@pytest.fixture(scope="module")
def pki(tmp_path_factory):
    """A test CA and server certificates: pinned.test, other.test, and one certificate valid for two names."""
    work = tmp_path_factory.mktemp("pki")

    def openssl(*args):
        subprocess.run([OPENSSL, *args], cwd=work, check=True, capture_output=True, timeout=60)

    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-keyout", "ca.key", "-out", "ca.pem", "-days", "1",
            "-subj", "/CN=webrag test CA", "-addext", "basicConstraints=critical,CA:TRUE",
            "-addext", "keyUsage=critical,keyCertSign")
    for name, sans in (("pinned", "DNS:pinned.test"), ("other", "DNS:other.test"), ("multi", "DNS:pinned.test,DNS:alias.test")):
        openssl("req", "-newkey", "rsa:2048", "-nodes", "-keyout", f"{name}.key", "-out", f"{name}.csr", "-subj", f"/CN={name}")
        (work / f"{name}.ext").write_text(f"subjectAltName={sans}\nextendedKeyUsage=serverAuth\n")
        openssl("x509", "-req", "-in", f"{name}.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
                "-out", f"{name}.pem", "-days", "1", "-extfile", f"{name}.ext")
    return work


# ------------------------------------------------------------------------------------------------ loopback servers
class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"                    # keep-alive, so pooled connections can be reused

    def do_GET(self):
        self.server.seen.append({"host": self.headers.get("Host"), "path": self.path})
        if self.path.startswith("/redirect") and self.server.redirect:
            self.send_response(302)
            self.send_header("Location", self.server.redirect)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        body = b"ok"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, cert=None, pki=None):
        super().__init__(("127.0.0.1", 0), Handler)
        self.seen, self.sni, self.redirect = [], [], ""
        if cert:
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(pki / f"{cert}.pem", pki / f"{cert}.key")
            context.sni_callback = lambda sock, name, ctx: self.sni.append(name)
            self.socket = context.wrap_socket(self.socket, server_side=True)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def port(self):
        return self.server_address[1]

    def handle_error(self, request, client_address):
        pass                                         # failed handshakes are expected in some tests


@pytest.fixture
def servers(pki):
    started = []

    def start(cert=None):
        server = Server(cert, pki)
        started.append(server)
        return server

    yield start
    for server in started:
        server.shutdown()
        server.server_close()


# ------------------------------------------------------------------------------------------------ environment
@pytest.fixture(autouse=True)
def offline(monkeypatch):
    """No proxies; 127.0.0.0/8 counts as public for the policy (and nothing else changes); every getaddrinfo call is
    recorded and ``*.test`` names cannot be resolved by the system."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "NO_PROXY",
                 "no_proxy", "WEBRAG_ALLOW_PROXY", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(requests.utils, "get_environ_proxies", lambda url, no_proxy=None: {})
    real_policy = net._blocked_address
    monkeypatch.setattr(net, "_blocked_address", lambda a: False if a in ipaddress.ip_network("127.0.0.0/8") else real_policy(a))
    real_getaddrinfo, lookups = socket.getaddrinfo, []

    def watching_getaddrinfo(host, *args, **kwargs):
        lookups.append(host.decode() if isinstance(host, bytes) else str(host))
        if lookups[-1].endswith(".test"):
            raise socket.gaierror(socket.EAI_NONAME, "test names are never resolved by the system")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", watching_getaddrinfo)
    return lookups


def make_session(pki, answers, **kwargs):
    """A PoliteSession that trusts the test CA; ``answers`` maps host -> list of addresses (or a callable)."""
    asked = []

    def resolver(host):
        asked.append(host)
        answer = answers[host]
        return answer() if callable(answer) else answer

    session = PoliteSession(min_interval=0, respect_robots=False, resolver=resolver, timeout=5, **kwargs)
    session.session.verify = str(pki / "ca.pem")
    session.asked = asked
    return session


def name_lookups(lookups):
    return [h for h in lookups if h.endswith(".test")]


def address_attempts(lookups):
    """urllib3 calls getaddrinfo on the numeric address it connects to; one call per connection attempt."""
    return [h for h in lookups if h.startswith("127.")]


# ------------------------------------------------------------------------------------------------ tests
def test_the_session_mounts_the_pinning_adapter_for_both_schemes():
    session = PoliteSession()
    assert isinstance(session.session.get_adapter("https://example.org/"), PinnedAdapter)
    assert isinstance(session.session.get_adapter("http://example.org/"), PinnedAdapter)


def test_a_pinned_https_request_verifies_tls_and_sends_the_original_name(pki, servers, offline):
    server = servers("pinned")
    session = make_session(pki, {"pinned.test": ["127.0.0.1"]})
    page = session.get(f"https://pinned.test:{server.port}/a")
    assert page.status == 200 and page.text == "ok"
    assert server.sni == ["pinned.test"]
    assert server.seen == [{"host": f"pinned.test:{server.port}", "path": "/a"}]
    assert page.url == f"https://pinned.test:{server.port}/a"
    assert name_lookups(offline) == []                # the name was never resolved again


def test_a_pinned_http_request_sends_the_original_host(pki, servers, offline):
    server = servers()
    session = make_session(pki, {"plain.test": ["127.0.0.1"]})
    assert session.get(f"http://plain.test:{server.port}/p").status == 200
    assert server.seen == [{"host": f"plain.test:{server.port}", "path": "/p"}]
    assert name_lookups(offline) == []


def test_a_certificate_for_another_name_fails_during_the_handshake(pki, servers, offline):
    server = servers("other")                         # valid chain, but for other.test
    session = make_session(pki, {"pinned.test": ["127.0.0.1"]})
    with pytest.raises(FetchError) as exc:
        session.get(f"https://pinned.test:{server.port}/a")
    assert exc.value.code == "tls" and server.seen == []      # no HTTP request reached the server


def test_an_untrusted_certificate_authority_fails(pki, servers):
    server = servers("pinned")
    session = make_session(pki, {"pinned.test": ["127.0.0.1"]})
    session.session.verify = True                     # the default bundle does not contain the test CA
    with pytest.raises(FetchError) as exc:
        session.get(f"https://pinned.test:{server.port}/a")
    assert exc.value.code == "tls" and server.seen == []


def test_turning_verification_off_or_client_certificates_are_refused(pki, servers):
    server = servers("pinned")
    session = make_session(pki, {"pinned.test": ["127.0.0.1"]})
    session.session.verify = False
    with pytest.raises(FetchError, match="may not be turned off") as exc:
        session.get(f"https://pinned.test:{server.port}/a")
    assert exc.value.code == "tls"
    session = make_session(pki, {"pinned.test": ["127.0.0.1"]})
    session.session.cert = str(pki / "pinned.pem")
    with pytest.raises(FetchError, match="client certificates") as exc:
        session.get(f"https://pinned.test:{server.port}/a")
    assert exc.value.code == "tls" and server.seen == [] and server.sni == []


def test_the_adapter_refuses_a_request_without_a_checked_address(pki, servers):
    server = servers("pinned")
    session = make_session(pki, {})
    with pytest.raises(PinningError) as exc:          # the requests session used directly, bypassing the guard
        session.session.get(f"https://pinned.test:{server.port}/a")
    assert exc.value.code == "blocked_host" and server.sni == []
    session._pin.value = ("another.test", ("127.0.0.1",))      # a pin for a different host does not apply
    with pytest.raises(PinningError):
        session.session.get(f"https://pinned.test:{server.port}/a")
    assert server.sni == []


def test_a_changed_dns_answer_does_not_redirect_the_connection(pki, servers, offline, monkeypatch):
    """DNS rebinding: the guard checked 127.0.0.1; later answers point at a private address. Within the reuse window
    the connection stays on the checked address; once it expires the new answer is checked and refused, and nothing
    is sent."""
    server = servers("pinned")
    answers = iter([["127.0.0.1"], ["10.0.0.9"]])
    session = make_session(pki, {"pinned.test": lambda: next(answers)})
    url = f"https://pinned.test:{server.port}/a"
    assert session.get(url).status == 200
    assert session.get(url).status == 200
    assert session.asked == ["pinned.test"] and len(server.seen) == 2
    monkeypatch.setattr(net, "ADDRESS_TTL", -1.0)     # the reuse window has passed
    with pytest.raises(FetchError) as exc:
        session.get(url)
    assert exc.value.code == "blocked_host" and len(server.seen) == 2
    assert name_lookups(offline) == []


def test_every_redirect_hop_is_checked_and_pinned_to_its_own_address(pki, servers, offline):
    second = servers("other")
    first = servers("pinned")
    first.redirect = f"https://other.test:{second.port}/landing"
    session = make_session(pki, {"pinned.test": ["127.0.0.1"], "other.test": ["127.0.0.1"]})
    page = session.get(f"https://pinned.test:{first.port}/redirect")
    assert page.status == 200 and page.url == f"https://other.test:{second.port}/landing"
    assert session.asked == ["pinned.test", "other.test"]
    assert first.sni == ["pinned.test"] and second.sni == ["other.test"]
    assert second.seen == [{"host": f"other.test:{second.port}", "path": "/landing"}]
    assert name_lookups(offline) == []


def test_a_redirect_to_a_host_with_a_private_answer_is_refused(pki, servers):
    target = servers("other")
    first = servers("pinned")
    first.redirect = f"https://other.test:{target.port}/landing"
    session = make_session(pki, {"pinned.test": ["127.0.0.1"], "other.test": ["10.1.2.3"]})
    with pytest.raises(FetchError) as exc:
        session.get(f"https://pinned.test:{first.port}/redirect")
    assert exc.value.code == "blocked_host" and target.seen == [] and target.sni == []


def test_connections_are_not_shared_between_host_names(pki, servers):
    server = servers("multi")                         # one server, one address, a certificate for both names
    session = make_session(pki, {"pinned.test": ["127.0.0.1"], "alias.test": ["127.0.0.1"]})
    session.get(f"https://pinned.test:{server.port}/1")
    session.get(f"https://alias.test:{server.port}/2")
    session.get(f"https://pinned.test:{server.port}/3")       # reuses the first name's pooled connection
    assert server.sni == ["pinned.test", "alias.test"]         # a separate handshake for the second name
    assert [s["host"] for s in server.seen] == [f"pinned.test:{server.port}", f"alias.test:{server.port}",
                                                f"pinned.test:{server.port}"]


def test_a_failed_connection_moves_to_the_next_checked_address(pki, servers, offline):
    server = servers("pinned")                        # listens on 127.0.0.1 only; 127.0.0.2 refuses the connection
    session = make_session(pki, {"pinned.test": ["127.0.0.2", "127.0.0.1"]})
    assert session.get(f"https://pinned.test:{server.port}/a").status == 200
    assert address_attempts(offline) == ["127.0.0.2", "127.0.0.1"]


def test_address_attempts_are_bounded(pki, servers, offline):
    server = servers("pinned")
    dead = [f"127.0.0.{n}" for n in range(2, 8)]      # six checked addresses, none listening
    session = make_session(pki, {"pinned.test": dead})
    with pytest.raises(FetchError) as exc:
        session.get(f"https://pinned.test:{server.port}/a")
    assert exc.value.code == "network"
    assert address_attempts(offline) == dead[:4]      # MAX_ADDRESS_ATTEMPTS


def test_a_tls_failure_is_not_retried_on_another_address(pki, servers, offline):
    server = servers("other")
    session = make_session(pki, {"pinned.test": ["127.0.0.1", "127.0.0.2"]})
    with pytest.raises(FetchError) as exc:
        session.get(f"https://pinned.test:{server.port}/a")
    assert exc.value.code == "tls" and address_attempts(offline) == ["127.0.0.1"]


def test_ipv4_answers_are_tried_before_ipv6(pki, servers, offline):
    server = servers("pinned")
    session = make_session(pki, {"pinned.test": ["2606:4700:4700::1111", "127.0.0.1"]})   # the IPv6 one is never used
    assert session.get(f"https://pinned.test:{server.port}/a").status == 200
    assert address_attempts(offline) == ["127.0.0.1"]


def test_an_accepted_proxy_receives_the_host_name_unpinned(pki, servers, offline, monkeypatch):
    """With WEBRAG_ALLOW_PROXY=1 the proxy resolves destinations itself (documented): the request goes to the proxy
    with the original name, and the adapter does not pin it."""
    proxy = servers()                                 # a plain HTTP server standing in for a forward proxy
    monkeypatch.setenv("WEBRAG_ALLOW_PROXY", "1")
    proxies = {"http": f"http://127.0.0.1:{proxy.port}"}
    monkeypatch.setattr(requests.utils, "get_environ_proxies", lambda url, no_proxy=None: proxies)
    session = make_session(pki, {"plain.test": ["127.0.0.1"]})
    session.session.proxies = proxies
    assert session.get("http://plain.test:8099/p").status == 200
    assert proxy.seen == [{"host": "plain.test:8099", "path": "http://plain.test:8099/p"}]
