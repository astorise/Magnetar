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

use magnetar_runtime::CliBoundaryError;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// CLI-owned network access policy (§10/§21 "Keep network policy in CLI").
/// Deny by default: network access only happens when both this policy
/// allows it and the caller explicitly requests it (e.g. `magnetar run
/// --url <url>`).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum NetworkPolicy {
    #[default]
    Deny,
    AllowExplicit,
}

const IO_TIMEOUT: Duration = Duration::from_secs(5);

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
    let mut stream = TcpStream::connect((host.as_str(), port)).map_err(|error| {
        CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to connect to '{host}:{port}': {error}"),
        }
    })?;
    stream.set_read_timeout(Some(IO_TIMEOUT)).ok();
    stream.set_write_timeout(Some(IO_TIMEOUT)).ok();
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nUser-Agent: magnetar-cli\r\n\r\n"
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to send request to '{host}:{port}': {error}"),
        })?;
    let mut response = Vec::new();
    stream
        .read_to_end(&mut response)
        .map_err(|error| CliBoundaryError::CliNetworkDenied {
            reason: format!("failed to read response from '{host}:{port}': {error}"),
        })?;
    let response = String::from_utf8_lossy(&response);
    let body = response
        .split_once("\r\n\r\n")
        .map(|(_, body)| body)
        .unwrap_or(&response);
    Ok(body.to_string())
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

    /// §10/§29 "Test network stays in CLI": with explicit policy, a real
    /// TCP connection is made -- entirely from the CLI process -- to a
    /// local, test-owned HTTP server, and the response body reaches the
    /// caller as a plain `String` never touched by Runtime.
    #[test]
    fn allow_explicit_fetches_body_from_a_local_http_server() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let marker = "MAGNETAR_CLI_NETWORK_TEST_MARKER_a91f";
        let handle = thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let mut buf = [0u8; 1024];
            let _ = stream.read(&mut buf);
            let body = format!("hello {marker}");
            let response = format!(
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                body.len(),
                body
            );
            stream.write_all(response.as_bytes()).unwrap();
        });
        let url = format!("http://{addr}/");
        let body = fetch_url_context(&url, NetworkPolicy::AllowExplicit).unwrap();
        assert!(body.contains(marker));
        handle.join().unwrap();
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
}
