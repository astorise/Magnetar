//! CLI-owned network access (§10 "Network Access" in the change proposal).
//!
//! `magnetar-cli` MAY access network services according to CLI policy;
//! Runtime SHALL not perform arbitrary network operations -- already
//! asserted structurally by `magnetar_runtime::cli_boundary::reject_cli_owned_authority`,
//! which rejects the `"network-tool"` capability name. This module fetches
//! plain `http://` URLs (no TLS dependency is added by this change, so
//! `https://` is rejected with a structured error rather than silently
//! downgrading or failing unclearly) and returns the response body as a
//! plain `String` -- the CLI never hands Runtime a URL, a socket, or any
//! other network authority, only the resulting text as explicit prompt
//! context (see `commands::cmd_run`'s `--url` flag).
//!
//! This is deliberately separate from Model Artifact distribution: model
//! downloads remain governed by the validated distribution contract
//! (`ModelArtifactSource` / Model Artifact provenance), not by this ad hoc
//! retrieval-context fetcher (see the change proposal's "Network Access"
//! section: "Future model distribution sources remain governed by the
//! validated distribution contract, not arbitrary inference-time network
//! authority").
//!
//! #57 hardening: the original implementation read the whole response with
//! no size cap (a memory-exhaustion surface for whatever server the user
//! points at), never bounded `TcpStream::connect` itself, discarded
//! `set_read_timeout`/`set_write_timeout` failures via `.ok()`, ignored the
//! HTTP status line (a redirect or error body was returned as if it were
//! successful content), did not decode `Transfer-Encoding: chunked` (its
//! framing leaked into the returned text as garbage), and lossily replaced
//! non-UTF-8 bytes instead of reporting them. All of those are fixed below
//! without adding a dependency, matching this module's existing "no TLS
//! dependency is added by this change" constraint. Full redirect-following
//! is deliberately still not implemented: rejecting a non-2xx status
//! (redirects included) as a structured error is a correct, honest
//! response for this module's purpose (assembling prompt context, not
//! browsing) -- following a redirect chain safely needs its own depth
//! limit and cross-origin policy, real follow-up work if ever needed
//! rather than a silent addition here.

use magnetar_runtime::CliBoundaryError;
use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

/// CLI-owned network access policy (§10/§21 "Keep network policy in CLI").
/// Network access only ever happens when both this policy allows it and
/// the caller explicitly requests it (e.g. `magnetar run --url <url>`) --
/// a flag alone is never sufficient.
///
/// This type's own [`Default`] is [`Self::Deny`], but that is not what
/// ships: [`crate::config::CliConfig::default`] sets `network_policy` to
/// [`Self::AllowExplicit`], so `--url` works out of the box today (see
/// that type's doc comment for why, and #55 for the discrepancy between
/// this comment's previous "deny by default" framing and the actually
/// shipped, permissive default). `Self::Deny` is real and reachable --
/// once a persistent configuration mechanism exists, setting it there
/// disables `--url` regardless of the flag -- it is simply not selected by
/// anything today.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NetworkPolicy {
    #[default]
    Deny,
    AllowExplicit,
}

const IO_TIMEOUT: Duration = Duration::from_secs(5);

/// Caps the total bytes read from one response (headers plus body). Large
/// enough for ordinary page/API prompt context, small enough that a
/// server streaming indefinitely cannot exhaust process memory (#57: the
/// previous `read_to_end` had no cap at all).
const MAX_RESPONSE_BYTES: usize = 8 * 1024 * 1024;

/// Parses `http://host[:port]/path` into `(host, port, path)`. Only the
/// `http` scheme is supported (see module doc comment); anything else is a
/// structured [`CliBoundaryError::CliNetworkDenied`], never a panic.
fn parse_http_url(url: &str) -> Result<(String, u16, String), CliBoundaryError> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| CliBoundaryError::CliNetworkDenied {
            reason: format!(
                "unsupported URL scheme in '{url}': only http:// is supported (no TLS dependency)"
            ),
        })?;
    let (authority, path) = match rest.find('/') {
        Some(index) => (&rest[..index], &rest[index..]),
        None => (rest, "/"),
    };
    if authority.is_empty() {
        return Err(CliBoundaryError::CliNetworkDenied {
            reason: format!("missing host in URL '{url}'"),
        });
    }
    let (host, port) = match authority.rsplit_once(':') {
        Some((host, port_str)) => {
            let port = port_str
                .parse::<u16>()
                .map_err(|_| CliBoundaryError::CliNetworkDenied {
                    reason: format!("invalid port in URL '{url}'"),
                })?;
            (host.to_string(), port)
        }
        None => (authority.to_string(), 80),
    };
    Ok((host, port, path.to_string()))
}

/// Reads up to `max_bytes` from `stream` (EOF-terminated: the request
/// always sends `Connection: close`, so the peer closing the connection is
/// the normal end of a response), returning [`CliBoundaryError::CliNetworkDenied`]
/// if the peer keeps sending past the cap instead of growing the buffer
/// without bound.
fn read_bounded(stream: &mut TcpStream, max_bytes: usize) -> Result<Vec<u8>, CliBoundaryError> {
    let mut response = Vec::new();
    let mut chunk = [0u8; 8192];
    loop {
        let read = stream
            .read(&mut chunk)
            .map_err(|error| CliBoundaryError::CliNetworkDenied {
                reason: format!("failed to read response: {error}"),
            })?;
        if read == 0 {
            break;
        }
        if response.len() + read > max_bytes {
            return Err(CliBoundaryError::CliNetworkDenied {
                reason: format!("response exceeded the {max_bytes}-byte limit"),
            });
        }
        response.extend_from_slice(&chunk[..read]);
    }
    Ok(response)
}

/// `(status_code, headers, body)`, the parsed shape [`parse_status_and_headers`]
/// returns.
type ParsedResponse<'a> = (u16, Vec<(String, String)>, &'a [u8]);

/// Splits a raw response into `(status_code, headers_lowercased, body)`.
/// `headers_lowercased` keeps only lowercase header names (values kept
/// as-is) so callers can look one up case-insensitively without redoing
/// that normalization themselves.
fn parse_status_and_headers(raw: &[u8]) -> Result<ParsedResponse<'_>, CliBoundaryError> {
    let text_prefix_end = raw
        .windows(4)
        .position(|window| window == b"\r\n\r\n")
        .ok_or_else(|| CliBoundaryError::CliNetworkDenied {
            reason: "response has no header terminator".into(),
        })?;
    let head = std::str::from_utf8(&raw[..text_prefix_end]).map_err(|_| {
        CliBoundaryError::CliNetworkDenied {
            reason: "response headers were not valid utf-8".into(),
        }
    })?;
    let mut lines = head.split("\r\n");
    let status_line = lines
        .next()
        .ok_or_else(|| CliBoundaryError::CliNetworkDenied {
            reason: "response has no status line".into(),
        })?;
    let status_code = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| CliBoundaryError::CliNetworkDenied {
            reason: format!("could not parse status line '{status_line}'"),
        })?;
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_ascii_lowercase(), value.trim().to_string()))
        .collect();
    let body = &raw[text_prefix_end + 4..];
    Ok((status_code, headers, body))
}

/// Decodes a `Transfer-Encoding: chunked` body (RFC 7230 §4.1): each chunk
/// is a hex size line, `\r\n`, that many data bytes, `\r\n`, repeated until
/// a zero-size chunk (any trailer headers after it are ignored, matching
/// what this module already does with the main response's own headers).
/// `max_bytes` bounds the decoded output the same way [`read_bounded`]
/// bounds the raw read.
fn decode_chunked_body(body: &[u8], max_bytes: usize) -> Result<Vec<u8>, CliBoundaryError> {
    let malformed = || CliBoundaryError::CliNetworkDenied {
        reason: "malformed chunked response body".into(),
    };
    let mut decoded = Vec::new();
    let mut cursor = body;
    loop {
        let line_end = cursor
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(malformed)?;
        let size_line = std::str::from_utf8(&cursor[..line_end]).map_err(|_| malformed())?;
        // A chunk-extension (";name=value") may follow the size; only the
        // size itself is needed here.
        let size_text = size_line.split(';').next().unwrap_or(size_line).trim();
        let size = usize::from_str_radix(size_text, 16).map_err(|_| malformed())?;
        cursor = &cursor[line_end + 2..];
        if size == 0 {
            break;
        }
        if cursor.len() < size + 2 {
            return Err(malformed());
        }
        if decoded.len() + size > max_bytes {
            return Err(CliBoundaryError::CliNetworkDenied {
                reason: format!("response exceeded the {max_bytes}-byte limit"),
            });
        }
        decoded.extend_from_slice(&cursor[..size]);
        cursor = &cursor[size + 2..]; // skip the chunk's trailing CRLF
    }
    Ok(decoded)
}

/// Fetches `url`'s body over a minimal HTTP/1.1 GET when `policy` allows it.
/// Entirely CLI-side: Runtime never sees `url`, the socket, or any network
/// authority -- only the returned `String` may later be folded into a
/// prompt by the caller (§10/§15 "Assemble network retrieval context in
/// CLI").
pub fn fetch_url_context(url: &str, policy: NetworkPolicy) -> Result<String, CliBoundaryError> {
    if !matches!(policy, NetworkPolicy::AllowExplicit) {
        return Err(CliBoundaryError::CliNetworkDenied {
            reason: format!("network access denied by CLI policy for '{url}'"),
        });
    }
    let (host, port, path) = parse_http_url(url)?;
    // DNS resolution itself (`to_socket_addrs`) is not bounded by
    // `IO_TIMEOUT` -- the standard library gives no portable way to time
    // out a hostname lookup without a dependency -- but the connection
    // attempt against the resolved address now is, which the previous
    // plain `TcpStream::connect` (no timeout at all, relying only on the
    // OS default) was not.
    let address = (host.as_str(), port)
        .to_socket_addrs()
        .map_err(|error| CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to resolve '{host}:{port}': {error}"),
        })?
        .next()
        .ok_or_else(|| CliBoundaryError::CliNetworkDenied {
            reason: format!("'{host}:{port}' resolved to no address"),
        })?;
    let mut stream = TcpStream::connect_timeout(&address, IO_TIMEOUT).map_err(|error| {
        CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to connect to '{host}:{port}': {error}"),
        }
    })?;
    // Previously `.ok()`, discarding a failure and silently proceeding
    // with no timeout at all.
    stream.set_read_timeout(Some(IO_TIMEOUT)).map_err(|error| {
        CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to set read timeout: {error}"),
        }
    })?;
    stream
        .set_write_timeout(Some(IO_TIMEOUT))
        .map_err(|error| CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to set write timeout: {error}"),
        })?;
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nUser-Agent: magnetar-cli\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to send request to '{host}:{port}': {error}"),
        })?;
    let raw = read_bounded(&mut stream, MAX_RESPONSE_BYTES)?;
    let (status_code, headers, body) = parse_status_and_headers(&raw)?;
    if !(200..300).contains(&status_code) {
        return Err(CliBoundaryError::CliNetworkDenied {
            reason: format!("'{url}' returned non-success status {status_code}"),
        });
    }
    let is_chunked = headers
        .iter()
        .any(|(name, value)| name == "transfer-encoding" && value.eq_ignore_ascii_case("chunked"));
    let decoded_body = if is_chunked {
        decode_chunked_body(body, MAX_RESPONSE_BYTES)?
    } else {
        body.to_vec()
    };
    String::from_utf8(decoded_body).map_err(|_| CliBoundaryError::CliNetworkDenied {
        reason: format!("'{url}' response body was not valid utf-8"),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;
    use std::thread;

    /// §10/§29 "Test network stays in CLI": deny is the default and never
    /// opens a socket, regardless of the URL.
    #[test]
    fn deny_policy_never_performs_network_io() {
        let error = fetch_url_context("http://127.0.0.1:1/", NetworkPolicy::Deny).unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliNetworkDenied { .. }));
    }

    #[test]
    fn https_scheme_is_rejected_without_connecting() {
        let error = fetch_url_context("https://example.invalid/", NetworkPolicy::AllowExplicit)
            .unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliNetworkDenied { .. }));
    }

    /// Spawns a local server that sends exactly `response` bytes to the
    /// first connection it accepts, then joins the request thread.
    fn serve_once(response: &'static str) -> std::net::SocketAddr {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            stream.write_all(response.as_bytes()).unwrap();
        });
        addr
    }

    /// §10/§29 "Test network stays in CLI": with explicit policy, a real
    /// TCP connection is made -- entirely from the CLI process -- to a
    /// local, test-owned HTTP server, and the response body reaches the
    /// caller as a plain `String` never touched by Runtime.
    #[test]
    fn allow_explicit_fetches_body_from_a_local_http_server() {
        let marker = "MAGNETAR_CLI_NETWORK_TEST_MARKER_a91f";
        let body = format!("hello {marker}");
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let addr = serve_once(Box::leak(response.into_boxed_str()));
        let url = format!("http://{addr}/");
        let fetched = fetch_url_context(&url, NetworkPolicy::AllowExplicit).unwrap();
        assert!(fetched.contains(marker));
    }

    /// #57: a non-2xx status (a redirect included) is rejected as an
    /// error, never returned as if it were the requested content.
    #[test]
    fn non_success_status_is_rejected_instead_of_returned_as_content() {
        let response = "HTTP/1.1 302 Found\r\nLocation: http://example.test/elsewhere\r\nConnection: close\r\n\r\n";
        let addr = serve_once(response);
        let url = format!("http://{addr}/");
        let error = fetch_url_context(&url, NetworkPolicy::AllowExplicit).unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliNetworkDenied { .. }));
    }

    /// #57: `Transfer-Encoding: chunked` is decoded rather than leaking its
    /// framing (chunk-size lines, trailing CRLFs) into the returned text.
    #[test]
    fn chunked_response_body_is_decoded() {
        let marker = "MAGNETAR_CLI_NETWORK_CHUNKED_MARKER_7e2b";
        let chunk_one = "hello ";
        let chunk_two = marker;
        let response = format!(
            "HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nConnection: close\r\n\r\n{:x}\r\n{}\r\n{:x}\r\n{}\r\n0\r\n\r\n",
            chunk_one.len(),
            chunk_one,
            chunk_two.len(),
            chunk_two,
        );
        let addr = serve_once(Box::leak(response.into_boxed_str()));
        let url = format!("http://{addr}/");
        let fetched = fetch_url_context(&url, NetworkPolicy::AllowExplicit).unwrap();
        assert_eq!(fetched, format!("{chunk_one}{chunk_two}"));
    }

    /// #57: a response larger than the cap is rejected rather than
    /// buffered without bound.
    #[test]
    fn response_over_the_size_cap_is_rejected() {
        let oversized_body = "x".repeat(MAX_RESPONSE_BYTES + 1);
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            oversized_body.len(),
            oversized_body
        );
        let addr = serve_once(Box::leak(response.into_boxed_str()));
        let url = format!("http://{addr}/");
        let error = fetch_url_context(&url, NetworkPolicy::AllowExplicit).unwrap_err();
        assert!(matches!(error, CliBoundaryError::CliNetworkDenied { .. }));
    }

    #[test]
    fn parse_http_url_splits_host_port_and_path() {
        assert_eq!(
            parse_http_url("http://example.test:8080/foo/bar").unwrap(),
            ("example.test".to_string(), 8080, "/foo/bar".to_string())
        );
        assert_eq!(
            parse_http_url("http://example.test").unwrap(),
            ("example.test".to_string(), 80, "/".to_string())
        );
    }

    #[test]
    fn parse_status_and_headers_extracts_status_code_and_lowercased_header_names() {
        let raw = b"HTTP/1.1 404 Not Found\r\nContent-Type: text/plain\r\n\r\nnot found";
        let (status, headers, body) = parse_status_and_headers(raw).unwrap();
        assert_eq!(status, 404);
        assert_eq!(
            headers,
            vec![("content-type".to_string(), "text/plain".to_string())]
        );
        assert_eq!(body, b"not found");
    }

    #[test]
    fn decode_chunked_body_concatenates_chunks_in_order() {
        let body = b"5\r\nhello\r\n1\r\n \r\n5\r\nworld\r\n0\r\n\r\n";
        let decoded = decode_chunked_body(body, MAX_RESPONSE_BYTES).unwrap();
        assert_eq!(decoded, b"hello world");
    }

    #[test]
    fn decode_chunked_body_rejects_malformed_framing() {
        assert!(decode_chunked_body(b"not-hex\r\ndata\r\n0\r\n\r\n", MAX_RESPONSE_BYTES).is_err());
    }
}
