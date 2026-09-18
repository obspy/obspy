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
from requests import PreparedRequest

from obspy.clients.base import HTTPClient
from obspy.clients.hooks import (
    BearerTokenHook, LoggingHook, RequestHookError, RequestHookHandler,
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

    @pytest.mark.parametrize("make_hook", [
        # The single-callable form...
        lambda resolver: BearerTokenHook(resolver),
        # ...and the dict-value-callable (per-host provider) form. Both
        # reach BearerTokenHook.__call__() through a different branch of
        # __init__(), so both are worth covering here even though the
        # resulting assertions are identical.
        lambda resolver: BearerTokenHook({"example.com": resolver}),
    ], ids=["single-callable", "dict-value-callable"])
    def test_callable_resolver_reevaluated_per_request(self, make_hook):
        # The whole point of a resolver/provider: a token that can be
        # refreshed, so its value must never be cached between requests.
        calls = []

        def resolver(request):
            calls.append(request.host)
            return "tok-%d" % len(calls)

        hook = make_hook(resolver)
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

    def test_dict_value_callable_is_invoked_for_its_host(self):
        hook = BearerTokenHook({"example.com": lambda request: "tok"})
        req = urllib_request.Request(URL)
        hook(_UrllibHookRequest(req))
        assert dict(req.header_items())["Authorization"] == "Bearer tok"

    def test_dict_mixes_static_and_callable_values(self):
        hook = BearerTokenHook({
            "example.com": lambda request: "dynamic",
            "other.org": "static",
        })
        dynamic = urllib_request.Request(URL)
        hook(_UrllibHookRequest(dynamic))
        static = urllib_request.Request("https://other.org/query")
        hook(_UrllibHookRequest(static))
        assert dict(dynamic.header_items())["Authorization"] == \
            "Bearer dynamic"
        assert dict(static.header_items())["Authorization"] == \
            "Bearer static"

    def test_dict_value_callable_returning_none_fails_open(self):
        hook = BearerTokenHook({"example.com": lambda request: None})
        req = urllib_request.Request(URL)
        hook(_UrllibHookRequest(req))
        assert "Authorization" not in dict(req.header_items())

    def test_dict_value_callable_not_invoked_when_https_required(self):
        # require_https is checked before any token is resolved, so a
        # plain-http request never triggers a token refresh.
        calls = []

        def provider(request):
            calls.append(request.host)
            return "tok"

        hook = BearerTokenHook({"example.com": provider})
        req = urllib_request.Request("http://example.com/query")
        hook(_UrllibHookRequest(req))
        assert calls == []
        assert "Authorization" not in dict(req.header_items())

    def test_dict_value_callable_not_invoked_for_another_host(self):
        # A provider is scoped to its own key: a request to a different
        # host must not even consult it, let alone carry its token.
        calls = []

        def provider(request):
            calls.append(request.host)
            return "tok"

        hook = BearerTokenHook({"other.org": provider})
        req = urllib_request.Request(URL)  # host: example.com
        hook(_UrllibHookRequest(req))
        assert calls == []
        assert "Authorization" not in dict(req.header_items())

    def test_dict_value_callable_error_names_the_hook_and_url(self):
        # A provider that raises (e.g. the auth SDK, when the user is not
        # logged in) must surface as RequestHookError keeping the
        # original exception as __cause__, and must say which hook and
        # which URL.
        def provider(request):
            raise ValueError("not logged in")

        hook = BearerTokenHook({"example.com": provider})
        req = _UrllibHookRequest(urllib_request.Request(URL))
        with pytest.raises(RequestHookError) as excinfo:
            _call_hook(hook, req, URL)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert "BearerTokenHook" in str(excinfo.value)
        assert URL in str(excinfo.value)

    def test_repr_names_hosts_without_leaking_tokens(self):
        # repr() reaches error messages and tracebacks (see _call_hook),
        # so it must name what the hook is scoped to and nothing more.
        hook = BearerTokenHook({"example.com": "s3cret", "other.org": "t0k"})
        text = repr(hook)
        assert "BearerTokenHook" in text
        assert "example.com" in text and "other.org" in text
        assert "s3cret" not in text and "t0k" not in text


class TestLoggingHookAndChain():
    def test_logging_hook_logs_and_does_not_mutate(self, caplog):
        req = urllib_request.Request(URL)
        before = dict(req.header_items())
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            LoggingHook()(_UrllibHookRequest(req))
        assert dict(req.header_items()) == before
        messages = [r.message for r in caplog.records]
        assert any("GET" in m and URL in m for m in messages)

    def test_logging_hook_default_omits_headers(self, caplog):
        req = urllib_request.Request(URL)
        req.add_header("X-Test", "header-value")
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            LoggingHook()(_UrllibHookRequest(req))
        [message] = [r.message for r in caplog.records]
        assert "header-value" not in message

    def test_logging_hook_headers_true_lists_header_names(self, caplog):
        req = urllib_request.Request(URL)
        req.add_header("X-Test", "value")
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            LoggingHook(headers=True)(_UrllibHookRequest(req))
        [message] = [r.message for r in caplog.records]
        assert "X-test: value" in message

    def test_logging_hook_redacts_authorization_by_default(self, caplog):
        req = urllib_request.Request(URL)
        req.add_unredirected_header("Authorization", "Bearer tok")
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            LoggingHook(headers=True)(_UrllibHookRequest(req))
        [message] = [r.message for r in caplog.records]
        assert "Authorization: <redacted>" in message
        assert "tok" not in message

    def test_chain_combining_auth_and_logging_redacts_the_token(
            self, caplog):
        # This is the ordering a caller naturally writes - authenticate,
        # then log - and it must never put the token itself in the log.
        req = urllib_request.Request(URL)
        hook = chain(
            BearerTokenHook({"example.com": "tok"}),
            LoggingHook(headers=True),
        )
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            hook(_UrllibHookRequest(req))
        messages = [r.message for r in caplog.records]
        assert any("Authorization: <redacted>" in m for m in messages)
        assert not any("tok" in m for m in messages)

    def test_logging_hook_redact_empty_shows_real_value(self, caplog):
        req = urllib_request.Request(URL)
        req.add_unredirected_header("Authorization", "Bearer tok")
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            LoggingHook(headers=True, redact=())(_UrllibHookRequest(req))
        [message] = [r.message for r in caplog.records]
        assert "Authorization: Bearer tok" in message

    def test_logging_hook_redact_matches_case_insensitively_both_backends(
            self, caplog):
        ureq = urllib_request.Request(URL)
        ureq.add_header("X-Api-Key", "secret")
        preq = PreparedRequest()
        preq.prepare(method="GET", url=URL, headers={"X-Api-Key": "secret"})
        hook = LoggingHook(headers=True, redact=("x-api-key",))
        with caplog.at_level(logging.DEBUG, logger="obspy.clients.hooks"):
            hook(_UrllibHookRequest(ureq))
            hook(_RequestsHookRequest(preq))
        messages = [r.message for r in caplog.records]
        assert len(messages) == 2
        for message in messages:
            assert "<redacted>" in message
            assert "secret" not in message

    def test_logging_hook_headers_true_skipped_when_level_disabled(
            self, caplog):
        req = urllib_request.Request(URL)
        req.add_unredirected_header("Authorization", "Bearer tok")
        # Set the level above DEBUG so the hook's own isEnabledFor() guard
        # short-circuits before it ever touches header values.
        with caplog.at_level(logging.INFO, logger="obspy.clients.hooks"):
            LoggingHook(headers=True)(_UrllibHookRequest(req))
        assert caplog.records == []

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

    def test_set_request_hook(self):
        # Unlike fdsn.Client, there is no opener/handler to rebuild here -
        # _download() reads self._request_hook fresh on every call.
        client = _DummyHTTPClient()
        assert client._request_hook is None

        hook = mock.Mock()
        client.set_request_hook(hook)
        assert client._request_hook is hook
        with mock.patch("requests.get") as get_mock:
            get_mock.return_value = mock.Mock(status_code=200)
            client._download(URL)
        assert "auth" in get_mock.call_args.kwargs

        client.set_request_hook(None)
        assert client._request_hook is None
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

    def test_syngine_client_forwards_request_hook(self):
        # obspy.clients.syngine.Client is a real HTTPClient subclass (unlike
        # _DummyHTTPClient above) - this is the one test proving its
        # __init__ actually forwards request_hook to HTTPClient.__init__
        # rather than swallowing it, without requiring network access.
        from obspy.clients.syngine import Client as SyngineClient

        def hook(request):
            pass

        c = SyngineClient(request_hook=hook)
        assert c._request_hook is hook
