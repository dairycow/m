//! Minimal blocking HTTP/1.1 client with full socket control.
//!
//! Owning the socket (instead of using a client crate) is what makes two
//! things possible: instant cancellation mid-stream (shutdown() makes
//! llama-server abort generation) and surviving long first-token latencies
//! (a 100k-token prompt fill is ~35s of silence) via short read timeouts
//! polled against a cancel flag.

use std::io::{ErrorKind, Read, Write};
use std::net::{Shutdown, TcpStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use crate::error::{Error, Result};

const READ_TIMEOUT: Duration = Duration::from_millis(200);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, Clone)]
pub struct Url {
    pub https: bool,
    pub host: String,
    pub port: u16,
    pub path: String,
}

impl Url {
    pub fn parse(s: &str) -> Result<Url> {
        let (https, rest) = if let Some(r) = s.strip_prefix("https://") {
            (true, r)
        } else if let Some(r) = s.strip_prefix("http://") {
            (false, r)
        } else {
            return Err(Error::msg(format!("URL must start with http:// or https://: {s}")));
        };
        let (authority, path) = match rest.find('/') {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, "/"),
        };
        let (host, port) = match authority.rsplit_once(':') {
            Some((h, p)) => (
                h.to_string(),
                p.parse::<u16>().map_err(|_| Error::msg(format!("bad port in URL: {s}")))?,
            ),
            None => (authority.to_string(), if https { 443 } else { 80 }),
        };
        if host.is_empty() {
            return Err(Error::msg(format!("no host in URL: {s}")));
        }
        Ok(Url { https, host, port, path: path.to_string() })
    }

    /// Join a base URL (scheme://host[:port][/prefix]) with a path.
    pub fn join(base: &str, path: &str) -> Result<Url> {
        let mut u = Url::parse(base)?;
        let prefix = u.path.trim_end_matches('/');
        u.path = format!("{prefix}{path}");
        Ok(u)
    }
}

enum Transport {
    Plain(TcpStream),
    #[cfg(feature = "tls")]
    Tls(Box<rustls::StreamOwned<rustls::ClientConnection, TcpStream>>),
}

impl Transport {
    fn tcp(&self) -> &TcpStream {
        match self {
            Transport::Plain(s) => s,
            #[cfg(feature = "tls")]
            Transport::Tls(s) => &s.sock,
        }
    }
}

impl Read for Transport {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        match self {
            Transport::Plain(s) => s.read(buf),
            #[cfg(feature = "tls")]
            Transport::Tls(s) => s.read(buf),
        }
    }
}

impl Write for Transport {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        match self {
            Transport::Plain(s) => s.write(buf),
            #[cfg(feature = "tls")]
            Transport::Tls(s) => s.write(buf),
        }
    }
    fn flush(&mut self) -> std::io::Result<()> {
        match self {
            Transport::Plain(s) => s.flush(),
            #[cfg(feature = "tls")]
            Transport::Tls(s) => s.flush(),
        }
    }
}

#[cfg(feature = "tls")]
fn tls_config() -> Arc<rustls::ClientConfig> {
    use std::sync::OnceLock;
    static CFG: OnceLock<Arc<rustls::ClientConfig>> = OnceLock::new();
    CFG.get_or_init(|| {
        let roots = rustls::RootCertStore {
            roots: webpki_roots::TLS_SERVER_ROOTS.to_vec(),
        };
        Arc::new(
            rustls::ClientConfig::builder()
                .with_root_certificates(roots)
                .with_no_client_auth(),
        )
    })
    .clone()
}

fn connect(url: &Url) -> Result<Transport> {
    use std::net::ToSocketAddrs;
    let addrs: Vec<_> = (url.host.as_str(), url.port)
        .to_socket_addrs()
        .map_err(|e| Error::msg(format!("resolve {}: {e}", url.host)))?
        .collect();
    if addrs.is_empty() {
        return Err(Error::msg(format!("no address for {}", url.host)));
    }
    // Try every resolved address (e.g. ::1 then 127.0.0.1 for localhost).
    let mut tcp = None;
    let mut last_err = None;
    for addr in &addrs {
        match TcpStream::connect_timeout(addr, CONNECT_TIMEOUT) {
            Ok(s) => {
                tcp = Some(s);
                break;
            }
            Err(e) => last_err = Some(e),
        }
    }
    let tcp = tcp.ok_or_else(|| {
        Error::msg(format!(
            "connect {}:{}: {}",
            url.host,
            url.port,
            last_err.map(|e| e.to_string()).unwrap_or_default()
        ))
    })?;
    tcp.set_nodelay(true).ok();
    tcp.set_read_timeout(Some(READ_TIMEOUT)).ok();
    if url.https {
        #[cfg(feature = "tls")]
        {
            let name = rustls::pki_types::ServerName::try_from(url.host.clone())
                .map_err(|_| Error::msg(format!("invalid TLS server name: {}", url.host)))?;
            let conn = rustls::ClientConnection::new(tls_config(), name)
                .map_err(|e| Error::msg(format!("TLS setup: {e}")))?;
            Ok(Transport::Tls(Box::new(rustls::StreamOwned::new(conn, tcp))))
        }
        #[cfg(not(feature = "tls"))]
        {
            Err(Error::msg("this build has no TLS support (https URL)"))
        }
    } else {
        Ok(Transport::Plain(tcp))
    }
}

/// A response whose body can be consumed line-by-line (for SSE) or fully.
pub struct Response {
    pub status: u16,
    headers: Vec<(String, String)>,
    transport: Transport,
    /// Bytes read past the header block, not yet consumed.
    buf: Vec<u8>,
    pos: usize,
    body: BodyKind,
    cancel: Arc<AtomicBool>,
}

#[derive(Debug)]
enum BodyKind {
    /// Chunked transfer-encoding; `remaining` = bytes left in current chunk.
    Chunked { remaining: usize, done: bool },
    Length { remaining: usize },
    /// Read until EOF (Connection: close).
    Eof,
}

pub fn post_json(
    url: &Url,
    headers: &[(&str, &str)],
    body: &[u8],
    cancel: Arc<AtomicBool>,
) -> Result<Response> {
    let mut t = connect(url)?;
    let mut req = String::with_capacity(256);
    req.push_str(&format!("POST {} HTTP/1.1\r\n", url.path));
    req.push_str(&format!("Host: {}\r\n", url.host));
    req.push_str("Content-Type: application/json\r\n");
    req.push_str("Accept: text/event-stream, application/json\r\n");
    req.push_str("Connection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
    write_all_retry(&mut t, req.as_bytes(), &cancel)?;
    write_all_retry(&mut t, body, &cancel)?;
    t.flush().map_err(|e| Error::msg(format!("send request: {e}")))?;
    read_response_head(t, cancel)
}

/// Write, retrying on WouldBlock (TLS handshake reads can hit the read timeout).
fn write_all_retry(t: &mut Transport, mut buf: &[u8], cancel: &AtomicBool) -> Result<()> {
    while !buf.is_empty() {
        if cancel.load(Ordering::Relaxed) {
            return Err(Error::Cancelled);
        }
        match t.write(buf) {
            Ok(0) => return Err(Error::msg("connection closed while sending request")),
            Ok(n) => buf = &buf[n..],
            Err(e) if retryable(&e) => continue,
            Err(e) => return Err(Error::msg(format!("send request: {e}"))),
        }
    }
    Ok(())
}

fn retryable(e: &std::io::Error) -> bool {
    matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut | ErrorKind::Interrupted)
}

fn read_response_head(mut t: Transport, cancel: Arc<AtomicBool>) -> Result<Response> {
    // Read until \r\n\r\n.
    let mut head = Vec::with_capacity(1024);
    let mut byte_buf = [0u8; 2048];
    let (buf, pos) = loop {
        if cancel.load(Ordering::Relaxed) {
            t.tcp().shutdown(Shutdown::Both).ok();
            return Err(Error::Cancelled);
        }
        match t.read(&mut byte_buf) {
            Ok(0) => return Err(Error::msg("connection closed before response headers")),
            Ok(n) => {
                head.extend_from_slice(&byte_buf[..n]);
                if let Some(end) = find_header_end(&head) {
                    let rest = head.split_off(end);
                    break (rest, 0);
                }
                if head.len() > 64 * 1024 {
                    return Err(Error::msg("response headers too large"));
                }
            }
            Err(e) if retryable(&e) => continue,
            Err(e) => return Err(Error::msg(format!("read response: {e}"))),
        }
    };

    let head_str = String::from_utf8_lossy(&head);
    let mut lines = head_str.split("\r\n");
    let status_line = lines.next().unwrap_or("");
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| Error::msg(format!("bad status line: {status_line}")))?;
    let mut headers = Vec::new();
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            headers.push((k.trim().to_ascii_lowercase(), v.trim().to_string()));
        }
    }

    let body = if header(&headers, "transfer-encoding").is_some_and(|v| v.contains("chunked")) {
        BodyKind::Chunked { remaining: 0, done: false }
    } else if let Some(len) = header(&headers, "content-length").and_then(|v| v.parse().ok()) {
        BodyKind::Length { remaining: len }
    } else {
        BodyKind::Eof
    };

    Ok(Response { status, headers, transport: t, buf, pos, body, cancel })
}

fn find_header_end(buf: &[u8]) -> Option<usize> {
    buf.windows(4).position(|w| w == b"\r\n\r\n").map(|i| i + 4)
}

fn header<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
}

impl Response {
    pub fn header(&self, name: &str) -> Option<&str> {
        header(&self.headers, name)
    }

    /// Abort the connection. The server treats this as client disconnect.
    pub fn abort(&self) {
        self.transport.tcp().shutdown(Shutdown::Both).ok();
    }

    /// Read the entire (decoded) body as a string.
    pub fn read_to_string(&mut self) -> Result<String> {
        let mut out = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            match self.read_body(&mut chunk)? {
                0 => break,
                n => out.extend_from_slice(&chunk[..n]),
            }
        }
        Ok(String::from_utf8_lossy(&out).into_owned())
    }

    /// Next line of the decoded body (without trailing \r\n).
    /// Returns None at end of body. Blocks, polling the cancel flag.
    pub fn next_line(&mut self) -> Result<Option<String>> {
        let mut line = Vec::new();
        let mut byte = [0u8; 1];
        loop {
            // Serve from decoded stream one byte at a time; the underlying
            // reads are buffered so this stays cheap.
            match self.read_body(&mut byte)? {
                0 => {
                    if line.is_empty() {
                        return Ok(None);
                    }
                    return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
                }
                _ => {
                    if byte[0] == b'\n' {
                        if line.last() == Some(&b'\r') {
                            line.pop();
                        }
                        return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
                    }
                    line.push(byte[0]);
                }
            }
        }
    }

    /// Decoded body read (dechunked). Returns 0 at end of body.
    fn read_body(&mut self, out: &mut [u8]) -> Result<usize> {
        loop {
            match self.body {
                BodyKind::Eof => {
                    if let Some(n) = self.serve_buffered(out, usize::MAX) {
                        return Ok(n);
                    }
                    match self.fill()? {
                        0 => return Ok(0),
                        _ => continue,
                    }
                }
                BodyKind::Length { remaining } => {
                    if remaining == 0 {
                        return Ok(0);
                    }
                    if let Some(n) = self.serve_buffered(out, remaining) {
                        if let BodyKind::Length { remaining } = &mut self.body {
                            *remaining -= n;
                        }
                        return Ok(n);
                    }
                    if self.fill()? == 0 {
                        return Ok(0);
                    }
                }
                BodyKind::Chunked { remaining, done } => {
                    if done {
                        return Ok(0);
                    }
                    if remaining == 0 {
                        // Parse next chunk-size line from buffered data.
                        match self.parse_chunk_header()? {
                            ParseChunk::NeedMore => {
                                if self.fill()? == 0 {
                                    // Truncated stream; treat as end.
                                    self.body = BodyKind::Chunked { remaining: 0, done: true };
                                    return Ok(0);
                                }
                            }
                            ParseChunk::Size(0) => {
                                self.body = BodyKind::Chunked { remaining: 0, done: true };
                                return Ok(0);
                            }
                            ParseChunk::Size(n) => {
                                self.body = BodyKind::Chunked { remaining: n, done: false };
                            }
                        }
                    } else {
                        if let Some(n) = self.serve_buffered(out, remaining) {
                            if let BodyKind::Chunked { remaining, .. } = &mut self.body {
                                *remaining -= n;
                            }
                            return Ok(n);
                        }
                        if self.fill()? == 0 {
                            return Ok(0);
                        }
                    }
                }
            }
        }
    }

    /// Copy up to `limit` buffered bytes into `out`. None if buffer is empty.
    fn serve_buffered(&mut self, out: &mut [u8], limit: usize) -> Option<usize> {
        let avail = self.buf.len() - self.pos;
        if avail == 0 {
            return None;
        }
        let n = avail.min(out.len()).min(limit);
        out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
        self.pos += n;
        if self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        }
        Some(n)
    }

    /// Try to parse a chunk-size line at the buffer head (skipping the CRLF
    /// that terminates the previous chunk).
    fn parse_chunk_header(&mut self) -> Result<ParseChunk> {
        let data = &self.buf[self.pos..];
        let mut i = 0;
        // Skip leading CRLF from the previous chunk's trailer.
        while i < data.len() && (data[i] == b'\r' || data[i] == b'\n') {
            i += 1;
        }
        let Some(nl) = data[i..].iter().position(|&b| b == b'\n') else {
            return Ok(ParseChunk::NeedMore);
        };
        let line = String::from_utf8_lossy(&data[i..i + nl]);
        let size_str = line.trim().split(';').next().unwrap_or("").trim();
        let size = usize::from_str_radix(size_str, 16)
            .map_err(|_| Error::msg(format!("bad chunk size: {size_str:?}")))?;
        self.pos += i + nl + 1;
        if self.pos == self.buf.len() {
            self.buf.clear();
            self.pos = 0;
        }
        Ok(ParseChunk::Size(size))
    }

    /// Read more bytes from the socket into the buffer, polling cancel on
    /// timeout. Returns number of bytes read (0 = clean EOF).
    fn fill(&mut self) -> Result<usize> {
        let mut chunk = [0u8; 16 * 1024];
        loop {
            if self.cancel.load(Ordering::Relaxed) {
                self.abort();
                return Err(Error::Cancelled);
            }
            match self.transport.read(&mut chunk) {
                Ok(0) => return Ok(0),
                Ok(n) => {
                    if self.pos > 0 && self.pos == self.buf.len() {
                        self.buf.clear();
                        self.pos = 0;
                    }
                    self.buf.extend_from_slice(&chunk[..n]);
                    return Ok(n);
                }
                Err(e) if retryable(&e) => continue,
                // rustls surfaces close_notify-less shutdown as UnexpectedEof.
                Err(e) if e.kind() == ErrorKind::UnexpectedEof => return Ok(0),
                Err(e) => return Err(Error::msg(format!("read body: {e}"))),
            }
        }
    }
}

enum ParseChunk {
    NeedMore,
    Size(usize),
}

/// Simple GET returning the body (for /props etc.). Not for streaming.
pub fn get_json(url: &Url, headers: &[(&str, &str)], cancel: Arc<AtomicBool>) -> Result<String> {
    let mut t = connect(url)?;
    let mut req = String::with_capacity(256);
    req.push_str(&format!("GET {} HTTP/1.1\r\n", url.path));
    req.push_str(&format!("Host: {}\r\n", url.host));
    req.push_str("Accept: application/json\r\n");
    req.push_str("Connection: close\r\n");
    for (k, v) in headers {
        req.push_str(&format!("{k}: {v}\r\n"));
    }
    req.push_str("\r\n");
    write_all_retry(&mut t, req.as_bytes(), &cancel)?;
    let mut resp = read_response_head(t, cancel)?;
    let body = resp.read_to_string()?;
    if resp.status >= 400 {
        return Err(Error::msg(format!("HTTP {}: {}", resp.status, truncate(&body, 300))));
    }
    Ok(body)
}

pub fn truncate(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_string()
    } else {
        let mut end = max;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}…", &s[..end])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_urls() {
        let u = Url::parse("http://localhost:8080").unwrap();
        assert!(!u.https);
        assert_eq!((u.host.as_str(), u.port, u.path.as_str()), ("localhost", 8080, "/"));
        let u = Url::parse("https://api.example.com/v1").unwrap();
        assert!(u.https);
        assert_eq!((u.port, u.path.as_str()), (443, "/v1"));
        let u = Url::join("http://localhost:8080", "/v1/chat/completions").unwrap();
        assert_eq!(u.path, "/v1/chat/completions");
        let u = Url::join("https://openrouter.ai/api/v1/", "/chat/completions").unwrap();
        assert_eq!(u.path, "/api/v1/chat/completions");
        assert!(Url::parse("localhost:8080").is_err());
    }
}
