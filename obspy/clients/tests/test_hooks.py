#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The obspy.clients.hooks test suite.

Focuses on the stack-neutral pieces shared by every client that supports a
``request_hook`` - the urllib and requests adapters seeing the same view of
a request, and the two backends wiring a hook in and wrapping its errors the
same way. FDSN-specific behaviour (BearerTokenHook scoped against real FDSN
URLs, redirect safety against CustomRedirectHandler, opener wiring on
:class:`~obspy.clients.fdsn.client.Client`) is covered in
``obspy.clients.fdsn.tests.test_client``.

:copyright:
    The ObsPy Development Team (devs@obspy.org)
:license:
    GNU Lesser General Public License, Version 3
    (https://www.gnu.org/copyleft/lesser.html)
"""
import logging
import urllib.request as urllib_request
from unittest import mock

import pytest
import requests
from requests import PreparedRequest

from obspy.clients.base import HTTPClient, RequestHookError
from obspy.clients.hooks import (
    BearerTokenHook, LoggingHook, RequestHookHandler,
    _call_hook, _RequestsHookRequest, _UrllibHookRequest, chain)


URL = "https://example.com/fdsnws/dataselect/1/query?net=IU"


class _DummyHTTPClient(HTTPClient):
    """
    Minimal concrete HTTPClient, only to exercise _download()'s hook
    wiring. Mirrors the pattern used for BaseRoutingClient in
    obspy.clients.fdsn.tests.test_base_routing_client.
    """
    def get_service_version(self):  # pragma: no cover
        return "0.0.0"

    def _handle_requests_http_error(self, r):  # pragma: no cover
        raise NotImplementedError


class TestHookRequestParity():
    """
    The same hook must see the same thing regardless of which backend
    (urllib or requests) is sending the request.
    """
    def setup_method(self):
        self.ureq = urllib_request.Request(URL)
        preq = PreparedRequest()
        preq.prepare(method="GET", url=URL, headers={})
        self.preq = preq
        self.uview = _UrllibHookRequest(self.ureq)
        self.rview = _RequestsHookRequest(self.preq)

    def test_read_only_attributes_match(self):
        for attr in ("url", "host", "scheme", "method"):
            assert getattr(self.uview, attr) == getattr(self.rview, attr), \
                attr

    def test_set_header_is_visible_on_both(self):
        # Header names are compared case-insensitively: urllib's
        # add_header() normalizes the name via str.capitalize() (so
        # "X-Test" becomes "X-test"), while requests preserves it as given.
        # Both are legal per HTTP - headers are case-insensitive.
        self.uview.set_header("X-Test", "value")
        self.rview.set_header("X-Test", "value")
        uheaders = {k.lower(): v for k, v in self.uview.headers.items()}
        rheaders = {k.lower(): v for k, v in self.rview.headers.items()}
        assert uheaders["x-test"] == "value"
        assert rheaders["x-test"] == "value"

    def test_set_secret_header_authorization_is_redirect_safe_on_both(self):
        # For "Authorization" specifically, set_secret_header() is
        # redirect-safe on both backends - see the docstring on
        # HookRequest.set_secret_header for why (add_unredirected_header()
        # on urllib; Session.rebuild_auth() stripping it on requests).
        self.uview.set_secret_header("Authorization", "Bearer tok")
        self.rview.set_secret_header("Authorization", "Bearer tok")

        # It is still sent...
        assert dict(self.ureq.header_items())["Authorization"] == \
            "Bearer tok"
        assert self.preq.headers["Authorization"] == "Bearer tok"

        # ...but on urllib it is not part of the plain header dict that a
        # redirect handler would copy onto a new request.
        assert "Authorization" not in self.ureq.headers


class TestBearerTokenHook():
    def test_dict_lookup_by_host(self):
        hook = BearerTokenHook({"example.com": "tok1", "other.org": "tok2"})
        req = urllib_request.Request(URL)
        hook(_UrllibHookRequest(req))
        assert dict(req.header_items())["Authorization"] == "Bearer tok1"

    def test_unmapped_host_fails_open(self):
        hook = BearerTokenHook({"other.org": "tok"})
        req = urllib_request.Request(URL)
        hook(_UrllibHookRequest(req))
        assert "Authorization" not in dict(req.header_items())

    def test_require_https_suppresses_header_on_http(self):
        hook = BearerTokenHook({"example.com": "tok"})
        req = urllib_request.Request("http://example.com/query")
        hook(_UrllibHookRequest(req))
        assert "Authorization" not in dict(req.header_items())

    def test_require_https_false_allows_http(self):
        hook = BearerTokenHook({"example.com": "tok"}, require_https=False)
        req = urllib_request.Request("http://example.com/query")
        hook(_UrllibHookRequest(req))
        assert dict(req.header_items())["Authorization"] == "Bearer tok"

    def test_callable_resolver_reevaluated_per_request(self):
        calls = []

        def resolver(request):
            calls.append(request.host)
            return "tok-%d" % len(calls)

        hook = BearerTokenHook(resolver)
        req1 = urllib_request.Request(URL)
        hook(_UrllibHookRequest(req1))
        req2 = urllib_request.Request(URL)
        hook(_UrllibHookRequest(req2))
        assert dict(req1.header_items())["Authorization"] == "Bearer tok-1"
        assert dict(req2.header_items())["Authorization"] == "Bearer tok-2"
        assert calls == ["example.com", "example.com"]

    def test_callable_resolver_none_means_no_header(self):
        hook = BearerTokenHook(lambda request: None)
        req = urllib_request.Request(URL)
        hook(_UrllibHookRequest(req))
        assert "Authorization" not in dict(req.header_items())

    def test_dict_lookup_is_case_insensitive(self):
        # Host names are not case-sensitive, so the mapping must not do an
        # exact string match on them.
        hook = BearerTokenHook({"Example.COM": "tok"})
        req = urllib_request.Request(URL)  # host: example.com
        hook(_UrllibHookRequest(req))
        assert dict(req.header_items())["Authorization"] == "Bearer tok"

    def test_dict_lookup_respects_explicit_port(self):
        hook = BearerTokenHook({"example.com:8080": "tok"})
        matching = urllib_request.Request("https://example.com:8080/query")
        hook(_UrllibHookRequest(matching))
        assert dict(matching.header_items())["Authorization"] == "Bearer tok"

        # Same host, no port in either the URL or the mapping key - and a
        # request to a *different* explicit port - must not match.
        other_port = urllib_request.Request("https://example.com:9090/query")
        hook(_UrllibHookRequest(other_port))
        assert "Authorization" not in dict(other_port.header_items())


class TestLoggingHookAndChain():
    def test_logging_hook_logs_and_does_not_mutate(self, caplog):
        req = urllib_request.Request(URL)
        before = dict(req.header_items())
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            LoggingHook()(_UrllibHookRequest(req))
        assert dict(req.header_items()) == before
        messages = [r.message for r in caplog.records]
        assert any("GET" in m and URL in m for m in messages)

    def test_chain_runs_hooks_in_order_and_shares_state(self):
        calls = []
        hook = chain(
            lambda request: calls.append("first"),
            lambda request: calls.append("second"),
        )
        hook(_UrllibHookRequest(urllib_request.Request(URL)))
        assert calls == ["first", "second"]

    def test_chain_combining_auth_and_logging(self):
        req = urllib_request.Request(URL)
        seen = []
        hook = chain(
            BearerTokenHook({"example.com": "tok"}),
            lambda request: seen.append(request.headers.get("Authorization")),
        )
        hook(_UrllibHookRequest(req))
        assert seen == ["Bearer tok"]

    def test_chain_error_names_the_failing_hook_not_itself(self):
        # A hook that raises inside a chain() must be named by the
        # resulting RequestHookError - not the wrapping `_chained` function
        # that chain() actually installs as the client's request_hook.
        def _broken(request):
            raise ValueError("boom")

        hook = chain(lambda request: None, _broken)
        with pytest.raises(RequestHookError) as excinfo:
            hook(_UrllibHookRequest(urllib_request.Request(URL)))
        assert repr(_broken) in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_chain_error_not_double_wrapped_by_outer_call_hook(self):
        # When a chain() is itself driven through _call_hook (as it is by
        # RequestHookHandler and the requests auth= wiring), a failure
        # inside one of its hooks must not be wrapped a second time - that
        # would again lose the identity of the hook that actually failed.
        def _broken(request):
            raise ValueError("boom")

        hook = chain(_broken)
        req = _UrllibHookRequest(urllib_request.Request(URL))
        with pytest.raises(RequestHookError) as excinfo:
            _call_hook(hook, req, URL)
        assert repr(_broken) in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)


class TestRequestHookHandler():
    def test_hook_invoked_and_request_returned(self):
        hook = mock.Mock()
        handler = RequestHookHandler(hook)
        req = urllib_request.Request(URL)
        returned = handler.http_request(req)
        assert returned is req
        assert hook.call_count == 1
        assert isinstance(hook.call_args.args[0], _UrllibHookRequest)

    def test_https_request_is_the_same_method(self):
        assert RequestHookHandler.https_request \
            is RequestHookHandler.http_request

    def test_hook_exception_wrapped_as_request_hook_error(self):
        def _broken(request):
            raise ValueError("boom")

        handler = RequestHookHandler(_broken)
        with pytest.raises(RequestHookError) as excinfo:
            handler.http_request(urllib_request.Request(URL))
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert URL in str(excinfo.value)


class TestHTTPClientRequestHook():
    """
    The requests-based backend (obspy.clients.base.HTTPClient), used by
    obspy.clients.syngine and the FDSN routing clients.
    """
    def test_download_passes_hook_through_requests_auth(self):
        hook = mock.Mock()
        client = _DummyHTTPClient(request_hook=hook)
        with mock.patch("requests.get") as get_mock:
            get_mock.return_value = mock.Mock(status_code=200)
            client._download(URL)
        auth = get_mock.call_args.kwargs["auth"]
        assert callable(auth)
        # Driving it the way requests would: call it with the (fake)
        # PreparedRequest and confirm it reaches our hook.
        fake_prepared = mock.Mock(url=URL)
        auth(fake_prepared)
        assert hook.call_count == 1
        assert isinstance(hook.call_args.args[0], _RequestsHookRequest)

    def test_no_hook_means_no_auth_kwarg(self):
        client = _DummyHTTPClient()
        with mock.patch("requests.get") as get_mock:
            get_mock.return_value = mock.Mock(status_code=200)
            client._download(URL)
        assert "auth" not in get_mock.call_args.kwargs

    def test_debug_url_printing_does_not_double_invoke_hook(self):
        hook = mock.Mock()
        client = _DummyHTTPClient(request_hook=hook, debug=True)
        with mock.patch("requests.get") as get_mock:
            get_mock.return_value = mock.Mock(status_code=200)
            client._download(URL)
        # requests.get() is mocked away entirely, so the only thing that
        # could have invoked the hook is the debug block's own throwaway
        # PreparedRequest.prepare() call, which must exclude "auth" from
        # the request_args it forwards - otherwise prepare_auth() would
        # call the hook a second, spurious time. This regresses loudly if
        # that exclusion is ever dropped from _download().
        assert hook.call_count == 0

    def test_broken_hook_raises_request_hook_error(self):
        def _broken(request):
            raise ValueError("boom")

        client = _DummyHTTPClient(request_hook=_broken)
        with mock.patch("requests.get") as get_mock:
            resp = mock.Mock(status_code=200)
            get_mock.return_value = resp

            def _fake_get(**kwargs):
                # Actually invoke the auth callable, the way requests does
                # internally while preparing the request.
                kwargs["auth"](mock.Mock(url=URL))
                return resp
            get_mock.side_effect = _fake_get

            with pytest.raises(RequestHookError) as excinfo:
                client._download(URL)
            assert isinstance(excinfo.value.__cause__, ValueError)


@pytest.mark.network
class TestHTTPClientRequestHookNetwork():
    """
    Same as above but exercising the real requests machinery end to end
    against an actual HTTP server, so the auth= wiring is proven against
    requests itself rather than only against a mock.
    """
    def test_bearer_token_hook_reaches_httpbin(self):
        hook = BearerTokenHook({"httpbin.org": "test-token"})
        client = _DummyHTTPClient(request_hook=hook)
        r = client._download("https://httpbin.org/headers")
        assert r.json()["headers"]["Authorization"] == "Bearer test-token"
        assert isinstance(r, requests.Response)
