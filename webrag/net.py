"""Polite HTTP access: timeouts, size caps, per-host spacing, robots.txt for page fetches, bounded retries on 429.

SSRF protection: every request - including every redirect hop, which is followed manually rather than by
``requests`` - must use http(s), must not name a private/loopback/link-local/reserved address, and must not resolve to
one. The only exception is an origin the operator configures explicitly (a self-hosted SearXNG on localhost), and
only for requests that start at that origin: a redirect from another origin into it is checked like any other hop.
An address is allowed only when it is globally routable (``is_global``) and none of the private, loopback, link-local,
reserved, multicast or unspecified flags apply; shared CGNAT space (100.64.0.0/10) is therefore refused. A NAT64
address in the RFC 6052 well-known prefix 64:ff9b::/96 (what DNS64 networks return for IPv4-only hosts) is judged by
the IPv4 address it embeds, so it is allowed exactly when that IPv4 destination would be; the local-use prefix
64:ff9b:1::/48 and other translation prefixes are not decoded and stay refused. Numeric host names in non-standard
forms (``2130706433``, ``0x7f.1``, ``127.1``) are refused before any lookup.
Proxies: a proxy taken from the environment or system settings resolves destinations itself, so the address check
could not cover the connection; such requests are refused unless WEBRAG_ALLOW_PROXY=1 accepts that trade-off.
Connection pinning (DNS rebinding): the addresses that passed the check are the only ones the connection may use.
``PinnedAdapter`` hands urllib3 the checked address instead of the host name, so the name is never resolved again,
while TLS keeps verifying the certificate chain and the original host name (sent as SNI) and the Host header carries
the original host. A connection failure moves to the next checked address, at most MAX_ADDRESS_ATTEMPTS in all; a TLS
failure is never retried. A checked answer is re-resolved after ADDRESS_TTL seconds. Requests through an accepted
proxy (WEBRAG_ALLOW_PROXY=1) are not pinned: the proxy resolves destinations itself.

Credentials: an origin is (scheme, host, effective port). When a redirect leaves the origin of the first request,
credential headers (``SENSITIVE_HEADERS``) are dropped for the rest of the chain, and query parameters are never
re-sent on any hop. URLs in errors and results are redacted (secret query values and user-info).
"""
from __future__ import annotations

import ipaddress
import os
import re
import socket
import threading
import time
import urllib.parse
import urllib.robotparser
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

import requests
import urllib3
from requests.adapters import HTTPAdapter
from requests.utils import select_proxy

USER_AGENT = "MultiAgentRAG-Research/0.1 (academic research prototype{contact})"
# Spacing per host. Wikipedia answered bursts at 0.5 s spacing with HTTP 429 during the first evaluation collection.
DEFAULT_HOST_INTERVAL = {"export.arxiv.org": 3.0, "api.semanticscholar.org": 1.5, "en.wikipedia.org": 1.0}
# Pauses up to this long are waited out (Wikipedia asked for 34 s in the eval collection); longer ones (OpenAlex's
# daily budget: 43560 s) fail fast and block the host until the pause ends.
MAX_RETRY_AFTER = 90.0
REDIRECT_CODES = {301, 302, 303, 307, 308}
UNREACHABLE = frozenset({"timeout", "network", "dns", "tls", "proxy"})   # failures to reach a host, not refusals
SECRET_PARAMS = re.compile(r"([?&](?:api_key|apikey|key|token|access_token)=)[^&#]*", re.I)


USERINFO = re.compile(r"^([A-Za-z][A-Za-z0-9+.\-]*://)[^/?#@]*@")
# Request headers that carry credentials; never sent to an origin other than the first request's.
SENSITIVE_HEADERS = frozenset({"authorization", "proxy-authorization", "cookie", "x-subscription-token",
                               "x-api-key", "api-key", "x-auth-token"})
DEFAULT_PORTS = {"http": 80, "https": 443}


def redact(url: str) -> str:
    """Remove secret query values and user-info (user:password@) so URLs can be stored or shown."""
    return SECRET_PARAMS.sub(r"\1REDACTED", USERINFO.sub(r"\1REDACTED@", url or ""))


def origin_of(url: "str | urllib.parse.ParseResult") -> str:
    """Canonical origin: scheme, lower-case host and effective port. Raises ValueError for a malformed URL."""
    parsed = urllib.parse.urlparse(url) if isinstance(url, str) else url
    scheme, host = parsed.scheme.lower(), (parsed.hostname or "").lower()
    if not scheme or not host:
        raise ValueError("URL without scheme or host")
    port = parsed.port if parsed.port is not None else DEFAULT_PORTS.get(scheme)    # .port raises if invalid
    return f"{scheme}://{f'[{host}]' if ':' in host else host}:{port}"


def _is_allowed(parsed: urllib.parse.ParseResult, allowed_origins: Iterable[str]) -> bool:
    if not allowed_origins:
        return False
    target = origin_of(parsed)
    for entry in allowed_origins:
        try:
            if origin_of(entry.rstrip("/")) == target:
                return True
        except ValueError:
            continue
    return False


def without_credentials(headers: Dict[str, str]) -> Dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in SENSITIVE_HEADERS}


NAT64_WELL_KNOWN = ipaddress.ip_network("64:ff9b::/96")      # RFC 6052; the IPv4 address is the last 32 bits
# Numeric host names: dotted, hexadecimal, octal or single-number forms that resolvers may read as an IPv4 address.
NUMERIC_HOST = re.compile(r"^(?:0x[0-9a-f]*|\d+)(?:\.(?:0x[0-9a-f]*|\d+)){0,3}$", re.I)


def nat64_embedded(address: ipaddress._BaseAddress) -> Optional[ipaddress.IPv4Address]:
    """The IPv4 address inside a well-known-prefix NAT64 address, or None. Only 64:ff9b::/96 is decoded: other
    prefixes (the local-use 64:ff9b:1::/48, operator prefixes) place the IPv4 bits differently by prefix length and
    are not trusted here."""
    if address.version == 6 and address in NAT64_WELL_KNOWN:
        return ipaddress.IPv4Address(int(address) & 0xFFFFFFFF)
    return None


def _blocked_address(address: ipaddress._BaseAddress) -> bool:
    embedded = nat64_embedded(address)
    if embedded is not None:                                  # reached through the network's NAT64 translator
        return _blocked_address(embedded)
    mapped = getattr(address, "ipv4_mapped", None)            # ::ffff:127.0.0.1 is 127.0.0.1
    if mapped is not None and _blocked_address(mapped):
        return True
    return (not address.is_global or address.is_private or address.is_loopback or address.is_link_local
            or address.is_reserved or address.is_multicast or address.is_unspecified)


def default_resolver(host: str) -> List[str]:
    return sorted({info[4][0] for info in socket.getaddrinfo(host, None)})


ADDRESS_TTL = 300.0           # seconds a checked DNS answer is reused before the host is resolved and checked again
MAX_ADDRESS_ATTEMPTS = 4      # checked addresses tried per request when connecting fails


class PinningError(requests.exceptions.RequestException):
    """Raised by PinnedAdapter before any connection is made; ``code`` is the FetchError code to report."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


def _connect_failed(exc: requests.exceptions.ConnectionError) -> bool:
    """True when the TCP connection itself could not be made (refused, unreachable, connect timeout): nothing was sent,
    so another checked address may be tried. TLS and proxy errors are not connect failures."""
    reason = getattr(exc.args[0] if exc.args else None, "reason", None)
    return isinstance(reason, (urllib3.exceptions.NewConnectionError, urllib3.exceptions.ConnectTimeoutError))


class PinnedAdapter(HTTPAdapter):
    """Connects to an address the guard checked for this request instead of resolving the host name again.

    ``pin()`` returns the (host, addresses) bound to the current thread's request by PoliteSession, or None. TLS still
    verifies the chain (requests' own CA settings) and the original host name: the name goes to urllib3 as
    ``server_hostname``, which OpenSSL uses for SNI and checks during the handshake. Pools are keyed by address and
    server_hostname, so connections are never shared between host names. Turning verification off and client
    certificates are refused."""

    def __init__(self, pin: Callable[[], Optional[Tuple[str, Tuple[str, ...]]]]):
        self._pin = pin
        super().__init__()

    def build_connection_pool_key_attributes(self, request, verify, cert=None):
        host_params, pool_kwargs = super().build_connection_pool_key_attributes(request, verify, cert)
        address = getattr(request, "pinned_address", None)
        if address is None:                             # an accepted proxy carries the request and resolves the name
            return host_params, pool_kwargs
        if pool_kwargs.get("cert_reqs") != "CERT_REQUIRED":
            raise PinningError("tls", "certificate verification may not be turned off")
        if "cert_file" in pool_kwargs:
            raise PinningError("tls", "client certificates are not supported")
        pool_kwargs["server_hostname"] = host_params["host"]
        host_params["host"] = address
        return host_params, pool_kwargs

    def send(self, request, stream=False, timeout=None, verify=True, cert=None, proxies=None):
        parsed = urllib.parse.urlsplit(request.url)
        if select_proxy(request.url, proxies):          # only reachable with WEBRAG_ALLOW_PROXY=1
            request.pinned_address = None
            return super().send(request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies)
        host, pin = (parsed.hostname or "").lower(), self._pin()
        if not pin or pin[0] != host or not pin[1]:
            raise PinningError("blocked_host", f"no checked address for {host}")
        request.headers["Host"] = parsed.netloc.rsplit("@", 1)[-1]
        addresses = pin[1][:MAX_ADDRESS_ATTEMPTS]
        for index, address in enumerate(addresses):
            request.pinned_address = address
            try:
                return super().send(request, stream=stream, timeout=timeout, verify=verify, cert=cert, proxies=proxies)
            except requests.exceptions.ConnectionError as exc:
                if index == len(addresses) - 1 or not _connect_failed(exc):
                    raise
        raise PinningError("blocked_host", f"no checked address for {host}")      # not reached


class FetchError(Exception):
    """``code``: invalid_url, blocked_host (refused by the address policy), proxy_policy (an environment proxy would
    carry the request), robots_disallowed, dns (the name could not be resolved), tls (certificate or handshake
    failure), proxy (the configured proxy failed), timeout, network, http_<status>, rate_limited, too_large,
    unsupported_type."""

    def __init__(self, code: str, message: str = ""):
        self.code = code
        super().__init__(f"{code}: {message}" if message else code)


@dataclass
class Fetched:
    url: str              # final URL after redirects
    status: int
    content_type: str
    content: bytes
    charset: str = ""     # from the Content-Type header, when the server declared one

    @property
    def text(self) -> str:
        """The body as text: byte-order mark, declared charset, HTML <meta charset>, UTF-8, then a best guess."""
        from .extract import decode_text
        return decode_text(self.content, self.charset, sniff_html=self.content_type in {"text/html", "application/xhtml+xml", ""})


def _check_url(url: str, allowed_origins: Iterable[str] = ()) -> urllib.parse.ParseResult:
    try:
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("not an http(s) URL with a host")
        origin_of(parsed)                                    # validates the port
    except ValueError:
        raise FetchError("invalid_url", redact(url)) from None
    if _is_allowed(parsed, allowed_origins):
        return parsed
    host = parsed.hostname.lower().rstrip(".")                # "localhost." is localhost
    if host == "localhost" or host.endswith(".local") or host.endswith(".localhost"):
        raise FetchError("blocked_host", host)
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if NUMERIC_HOST.match(host):                          # 2130706433, 0x7f.1, 127.1: not a standard address
            raise FetchError("blocked_host", f"{host} is a numeric address in a non-standard form") from None
        return parsed
    if _blocked_address(address):
        raise FetchError("blocked_host", host)
    return parsed


class PoliteSession:
    def __init__(self, timeout: float = 15.0, max_bytes: int = 8_000_000, min_interval: float = 0.5,
                 host_interval: Optional[Dict[str, float]] = None, respect_robots: bool = True, max_redirects: int = 5,
                 resolver: Callable[[str], List[str]] = default_resolver, allowed_origins: Iterable[str] = ()):
        contact = os.environ.get("WEBRAG_CONTACT", "").strip()
        self.user_agent = USER_AGENT.format(contact=f"; {contact}" if contact else "")
        self.timeout, self.max_bytes, self.min_interval = timeout, max_bytes, min_interval
        self.host_interval = dict(DEFAULT_HOST_INTERVAL, **(host_interval or {}))
        self.respect_robots = respect_robots
        self.max_redirects = max_redirects
        self.resolver = resolver
        # Operator-configured origins (e.g. a local SearXNG) that may be private; never derived from fetched content.
        self.allowed_origins: Set[str] = {o.rstrip("/").lower() for o in allowed_origins}
        # (host, operator-exempt) -> (time checked, addresses in connection order, or None when refused)
        self._checked: Dict[Tuple[str, bool], Tuple[float, Optional[Tuple[str, ...]]]] = {}
        self._pin = threading.local()                   # the checked (host, addresses) of this thread's request
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": self.user_agent})
        adapter = PinnedAdapter(lambda: getattr(self._pin, "value", None))
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
        self._lock = threading.Lock()
        self._host_locks: Dict[str, threading.Lock] = {}
        self._last: Dict[str, float] = {}
        self._robots: Dict[str, Optional[urllib.robotparser.RobotFileParser]] = {}
        self._blocked_until: Dict[str, float] = {}      # hosts that asked us to wait longer than MAX_RETRY_AFTER
        self.requests_made = 0

    def _proxied(self, url: str) -> bool:
        """Whether ``requests`` would send this URL through a proxy taken from the environment or system settings,
        when that has not been accepted with WEBRAG_ALLOW_PROXY=1."""
        if os.environ.get("WEBRAG_ALLOW_PROXY", "").strip() == "1" or not getattr(self.session, "trust_env", False):
            return False
        try:
            return bool(requests.utils.get_environ_proxies(url))
        except (ValueError, OSError):
            return True                                  # cannot tell: refuse rather than guess

    def _host_lock(self, host: str) -> threading.Lock:
        with self._lock:
            return self._host_locks.setdefault(host, threading.Lock())

    def _wait_turn(self, host: str) -> None:
        interval = self.host_interval.get(host, self.min_interval)
        wait = self._last.get(host, 0.0) + interval - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        self._last[host] = time.monotonic()

    def _robots_allows(self, parsed: urllib.parse.ParseResult) -> bool:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        if origin not in self._robots:
            parser: Optional[urllib.robotparser.RobotFileParser] = urllib.robotparser.RobotFileParser()
            try:
                response = self._raw_get(origin + "/robots.txt", check_robots=False, max_bytes=500_000)
                parser.parse(response.text.splitlines())
            except FetchError as exc:
                # No robots.txt (4xx) means no restrictions; an unreachable one is treated the same way, because
                # the page fetch itself will then fail and be reported. A policy refusal is never one of these.
                parser = None if exc.code.startswith("http_4") or exc.code in UNREACHABLE else parser
            self._robots[origin] = parser
        parser = self._robots[origin]
        return True if parser is None else parser.can_fetch(self.user_agent, parsed.geturl())

    def get(self, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None,
            check_robots: bool = True, max_bytes: Optional[int] = None) -> Fetched:
        """``check_robots`` is turned off only for documented public APIs (search endpoints), never for pages."""
        return self._raw_get(url, params=params, headers=headers, check_robots=check_robots, max_bytes=max_bytes)

    def _check_resolved(self, parsed: urllib.parse.ParseResult, allowed_origins: Iterable[str] = ()) -> Tuple[str, ...]:
        """The addresses the connection to this host may use: refuses public-looking host names that resolve to
        private addresses (one refused address refuses the host). IPv4 addresses come first, then IPv6 (including
        NAT64). An answer is reused for ADDRESS_TTL seconds. An operator-configured origin is resolved without the
        policy, and a literal address (checked by _check_url) is its own answer."""
        host = parsed.hostname.lower()
        try:
            ipaddress.ip_address(host)
            return (host,)
        except ValueError:
            pass
        exempt = _is_allowed(parsed, allowed_origins)
        cached = self._checked.get((host, exempt))
        if cached is None or time.monotonic() - cached[0] > ADDRESS_TTL:
            try:
                answer = [ipaddress.ip_address(a.split("%")[0]) for a in self.resolver(host)]
            except (OSError, UnicodeError, ValueError):
                raise FetchError("dns", f"cannot resolve {host}") from None
            refused = not answer or (not exempt and any(_blocked_address(a) for a in answer))
            ordered = tuple(str(a) for a in sorted(answer, key=lambda a: a.version))      # stable: IPv4 first
            cached = (time.monotonic(), None if refused else ordered)
            self._checked[(host, exempt)] = cached
        if cached[1] is None:
            raise FetchError("blocked_host", f"{host} resolves to a private or reserved address")
        return cached[1]

    def _send(self, url: str, host: str, params: Optional[dict], headers: Optional[dict],
              addresses: Tuple[str, ...] = ()) -> requests.Response:
        """One request (redirects not followed) with per-host spacing and the 429 policy, connected only to the
        checked ``addresses``."""
        remaining = self._blocked_until.get(host, 0.0) - time.monotonic()
        if remaining > 0:                            # honour a long Retry-After without contacting the host again
            raise FetchError("rate_limited", f"{host} asked for a pause; {int(remaining)} s remaining")
        if self._proxied(url):
            raise FetchError("proxy_policy", f"a proxy from the environment or system settings would carry the request "
                                             f"to {host}; set WEBRAG_ALLOW_PROXY=1 to allow it (the proxy then resolves "
                                             f"destinations itself, outside the address check)")
        for attempt in range(2):
            with self._host_lock(host):
                self._wait_turn(host)
                self.requests_made += 1
                self._pin.value = (host, tuple(addresses))
                try:
                    response = self.session.get(url, params=params, headers=headers, timeout=self.timeout,
                                                stream=True, allow_redirects=False)
                except PinningError as exc:
                    raise FetchError(exc.code, f"{exc}: {redact(url)}") from None
                except requests.exceptions.Timeout:
                    raise FetchError("timeout", redact(url)) from None
                except requests.exceptions.SSLError as exc:
                    raise FetchError("tls", f"{type(exc).__name__}: {redact(url)}") from None
                except requests.exceptions.ProxyError as exc:
                    raise FetchError("proxy", f"{type(exc).__name__}: {redact(url)}") from None
                except requests.exceptions.ConnectionError as exc:
                    code = "dns" if "NameResolution" in repr(exc) or "getaddrinfo" in repr(exc) else "network"
                    raise FetchError(code, f"{type(exc).__name__}: {redact(url)}") from None
                except requests.exceptions.RequestException as exc:
                    raise FetchError("network", f"{type(exc).__name__}: {redact(url)}") from None
                finally:
                    self._pin.value = None
            if response.status_code == 429 and attempt == 0:
                retry_after = response.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 10.0
                response.close()
                if delay > MAX_RETRY_AFTER:
                    self._blocked_until[host] = time.monotonic() + delay
                    raise FetchError("rate_limited", f"{host} asked to retry after {int(delay)} s")
                with self._host_lock(host):             # hold other requests to this host during the back-off
                    time.sleep(delay)
                continue
            if response.status_code == 429:
                response.close()
                raise FetchError("rate_limited", redact(url))
            return response
        raise FetchError("rate_limited", redact(url))

    def _raw_get(self, url: str, *, params: Optional[dict] = None, headers: Optional[dict] = None,
                 check_robots: bool = True, max_bytes: Optional[int] = None) -> Fetched:
        limit = max_bytes or self.max_bytes
        current, current_params = url, params
        send_headers = dict(headers) if headers else None
        first_origin: Optional[str] = None
        for _ in range(self.max_redirects + 1):
            # The operator-configured exemption covers only hops within the origin the request started at.
            try:
                same_origin = first_origin is None or origin_of(current) == first_origin
            except ValueError:
                same_origin = False                           # malformed; _check_url rejects it below
            exempt = self.allowed_origins if same_origin else ()
            parsed = _check_url(current, exempt)
            addresses = self._check_resolved(parsed, exempt)
            if first_origin is None:
                first_origin = origin_of(parsed)
            elif send_headers and not same_origin:
                send_headers = without_credentials(send_headers)       # dropped for the rest of the chain
            if check_robots and self.respect_robots and not self._robots_allows(parsed):
                raise FetchError("robots_disallowed", redact(current))
            response = self._send(current, parsed.hostname.lower(), current_params, send_headers, addresses)
            if response.status_code in REDIRECT_CODES:
                location = response.headers.get("Location", "")
                response.close()
                if not location:
                    raise FetchError("network", f"redirect without a location: {redact(current)}")
                # Every hop is re-checked above: scheme, private addresses, DNS answer and robots.txt.
                try:
                    current = urllib.parse.urljoin(response.url or current, location)
                except ValueError:
                    raise FetchError("invalid_url", f"malformed redirect location from {redact(current)}") from None
                current_params = None
                continue
            if response.status_code >= 400:
                response.close()
                raise FetchError(f"http_{response.status_code}", redact(current))
            declared = response.headers.get("Content-Length", "")
            if declared.isdigit() and int(declared) > limit:
                response.close()
                raise FetchError("too_large", f"{declared} bytes")
            body = bytearray()
            try:
                for block in response.iter_content(65536):
                    body.extend(block)
                    if len(body) > limit:
                        raise FetchError("too_large", f"over {limit} bytes")
            except requests.exceptions.RequestException as exc:
                raise FetchError("network", f"{type(exc).__name__} while reading {redact(current)}") from None
            finally:
                response.close()
            content_type = response.headers.get("Content-Type", "")
            charset = re.search(r"charset=([^;]+)", content_type, re.I)
            return Fetched(redact(response.url or current), response.status_code,
                           content_type.split(";")[0].strip().lower(), bytes(body),
                           charset.group(1).strip().strip('"') if charset else "")
        raise FetchError("network", f"too many redirects: {redact(url)}")
