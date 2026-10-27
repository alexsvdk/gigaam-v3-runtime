// SPDX-License-Identifier: LGPL-3.0-or-later
// OpenAI-compatible batch transcription adapter.
use crate::http::{self, HttpState, RequestId, TranscriptionResult};
use crate::json::JSON_CONTENT_TYPE;
use crate::json::{Json, obj};
use axum::{
    extract::{FromRequest, Multipart, Request, State},
    http::{StatusCode, header},
    response::{IntoResponse, Response},
};
use std::path::Path;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Format {
    Json,
    Text,
    Verbose,
}
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Engine {
    Ctc,
    Rnnt,
}
#[derive(Clone, Debug)]
struct Input {
    audio: Vec<u8>,
    extension: Option<String>,
    engine: Engine,
    format: Format,
    word: bool,
    segment: bool,
}
pub(crate) struct OpenAiWorkInput {
    pub(crate) extension: Option<String>,
    pub(crate) rnnt: bool,
}

#[derive(Debug)]
struct ApiError {
    status: StatusCode,
    message: String,
    param: Option<&'static str>,
}
impl ApiError {
    fn bad(message: impl Into<String>, param: Option<&'static str>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
            param,
        }
    }
    fn response(self) -> Response {
        (
            self.status,
            [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
            obj(vec![(
                "error",
                obj(vec![
                    ("message", Json::str(self.message)),
                    (
                        "type",
                        Json::str(if self.status.is_client_error() {
                            "invalid_request_error"
                        } else {
                            "server_error"
                        }),
                    ),
                    ("param", self.param.map(Json::str).unwrap_or(Json::Null)),
                    ("code", Json::Null),
                ]),
            )])
            .to_string(),
        )
            .into_response()
    }
}

pub(crate) fn error_response(
    status: StatusCode,
    message: impl Into<String>,
    param: Option<&'static str>,
) -> Response {
    ApiError {
        status,
        message: message.into(),
        param,
    }
    .response()
}

fn parse_model(s: &str, rnnt: bool) -> Result<Engine, ApiError> {
    match s {
        "gigaam-v3" | "gigaam-v3-ctc" => Ok(Engine::Ctc),
        "gigaam-v3-rnnt" if rnnt => Ok(Engine::Rnnt),
        "gigaam-v3-rnnt" => Err(ApiError::bad("RNN-T model is not loaded", Some("model"))),
        _ => Err(ApiError::bad(
            format!("Unsupported model: {s}"),
            Some("model"),
        )),
    }
}
fn extension(filename: Option<&str>) -> Option<String> {
    let base = filename
        .and_then(|s| Path::new(s).file_name())
        .and_then(|s| s.to_str())?;
    let ext = Path::new(base).extension()?.to_str()?.to_ascii_lowercase();
    [
        "wav", "flac", "ogg", "opus", "mp3", "aac", "m4a", "mp4", "mkv", "vorbis", "pcm",
    ]
    .contains(&ext.as_str())
    .then_some(ext)
}
fn parse_fields(
    mut fields: Vec<(String, Option<String>, Vec<u8>)>,
    rnnt: bool,
) -> Result<Input, ApiError> {
    let mut file = None;
    let mut model = None;
    let mut format = Format::Json;
    let mut fmt_seen = false;
    let mut word = false;
    let mut segment = false;
    let mut granularities = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for (name, filename, bytes) in fields.drain(..) {
        if name != "timestamp_granularities[]" && !seen.insert(name.clone()) {
            return Err(ApiError::bad(
                format!("{name} must occur once"),
                Some(match name.as_str() {
                    "file" => "file",
                    "model" => "model",
                    "response_format" => "response_format",
                    "language" => "language",
                    "temperature" => "temperature",
                    "stream" => "stream",
                    "prompt" => "prompt",
                    _ => "field",
                }),
            ));
        }
        match name.as_str() {
            "file" => {
                if file.is_some() {
                    return Err(ApiError::bad("file must occur once", Some("file")));
                }
                if bytes.is_empty() {
                    return Err(ApiError::bad("file must not be empty", Some("file")));
                }
                file = Some((bytes, extension(filename.as_deref())));
            }
            "model" => {
                if model.is_some() {
                    return Err(ApiError::bad("model must occur once", Some("model")));
                }
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("model must be UTF-8", Some("model")))?;
                if s.is_empty() {
                    return Err(ApiError::bad("model must not be empty", Some("model")));
                }
                model = Some(parse_model(&s, rnnt)?);
            }
            "response_format" => {
                if fmt_seen {
                    return Err(ApiError::bad(
                        "response_format must occur once",
                        Some("response_format"),
                    ));
                }
                fmt_seen = true;
                let s = String::from_utf8(bytes).map_err(|_| {
                    ApiError::bad("response_format must be UTF-8", Some("response_format"))
                })?;
                format = match s.as_str() {
                    "json" => Format::Json,
                    "text" => Format::Text,
                    "verbose_json" => Format::Verbose,
                    _ => {
                        return Err(ApiError::bad(
                            format!("Unsupported response_format: {s}"),
                            Some("response_format"),
                        ));
                    }
                }
            }
            "timestamp_granularities[]" => {
                let s = String::from_utf8(bytes).map_err(|_| {
                    ApiError::bad(
                        "timestamp_granularities[] must be UTF-8",
                        Some("timestamp_granularities[]"),
                    )
                })?;
                if granularities.contains(&s) {
                    return Err(ApiError::bad(
                        "duplicate timestamp granularity",
                        Some("timestamp_granularities[]"),
                    ));
                }
                match s.as_str() {
                    "word" => word = true,
                    "segment" => segment = true,
                    _ => {
                        return Err(ApiError::bad(
                            "timestamp granularity must be word or segment",
                            Some("timestamp_granularities[]"),
                        ));
                    }
                }
                granularities.push(s);
            }
            "language" => {
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("language must be UTF-8", Some("language")))?;
                if s != "ru" {
                    return Err(ApiError::bad(
                        "Only language=ru is supported",
                        Some("language"),
                    ));
                }
            }
            "temperature" => {
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("temperature must be UTF-8", Some("temperature")))?;
                if s.parse::<f64>()
                    .map_or(true, |value| !value.is_finite() || value != 0.0)
                {
                    return Err(ApiError::bad(
                        "Only temperature=0 is supported",
                        Some("temperature"),
                    ));
                }
            }
            "stream" => {
                let s = String::from_utf8(bytes)
                    .map_err(|_| ApiError::bad("stream must be UTF-8", Some("stream")))?;
                if s != "false" {
                    return Err(ApiError::bad("streaming is not supported", Some("stream")));
                }
            }
            "prompt" => {
                if !bytes.is_empty() {
                    return Err(ApiError::bad("prompt is not supported", Some("prompt")));
                }
            }
            _ => return Err(ApiError::bad(format!("Unsupported field: {name}"), None)),
        }
    }
    if format != Format::Verbose && !granularities.is_empty() {
        return Err(ApiError::bad(
            "timestamp granularities require verbose_json",
            Some("timestamp_granularities[]"),
        ));
    }
    let (audio, extension) = file.ok_or_else(|| ApiError::bad("file is required", Some("file")))?;
    let engine = model.ok_or_else(|| ApiError::bad("model is required", Some("model")))?;
    if format == Format::Verbose && granularities.is_empty() {
        segment = true;
    }
    Ok(Input {
        audio,
        extension,
        engine,
        format,
        word,
        segment,
    })
}

pub(crate) fn routes() -> axum::Router<HttpState> {
    axum::Router::new()
        .route("/v1/audio/transcriptions", axum::routing::post(transcribe))
        .route("/v1/models", axum::routing::get(models))
}

async fn transcribe(State(state): State<HttpState>, request: Request) -> Response {
    state.metrics.transcribe();
    let started = std::time::Instant::now();
    let (parts, body) = request.into_parts();
    let headers = parts.headers.clone();
    let upload_guard = parts.extensions.get::<http::UploadGuard>().cloned();
    let request_id = match RequestId::from_headers(&headers) {
        Ok(v) => v,
        Err(e) => return ApiError::bad(e, None).response(),
    };
    let request = Request::from_parts(parts, body);
    let mut multipart = match Multipart::from_request(request, &state).await {
        Ok(multipart) => multipart,
        Err(_) => {
            state.metrics.transcribe_error();
            return with_id(
                ApiError::bad("Expected multipart/form-data with a valid boundary", None)
                    .response(),
                &request_id,
            );
        }
    };
    let mut parts = Vec::new();
    let mut total = 0usize;
    let read_deadline = tokio::time::Instant::now() + state.request_timeout;
    loop {
        let next_field = match tokio::time::timeout_at(read_deadline, multipart.next_field()).await
        {
            Ok(result) => result,
            Err(_) => {
                state.metrics.timeout();
                return with_id(
                    ApiError {
                        status: StatusCode::GATEWAY_TIMEOUT,
                        message: "request body read timeout".into(),
                        param: None,
                    }
                    .response(),
                    &request_id,
                );
            }
        };
        let field = match next_field {
            Ok(Some(f)) => f,
            Ok(None) => break,
            Err(e) => {
                state.metrics.transcribe_error();
                let status = e.status();
                let mut error =
                    ApiError::bad(format!("Invalid multipart body: {}", e.body_text()), None);
                if status == StatusCode::PAYLOAD_TOO_LARGE {
                    error.status = status;
                    error.message = "request body too large".into();
                }
                return with_id(error.response(), &request_id);
            }
        };
        if parts.len() >= 16 {
            state.metrics.transcribe_error();
            return with_id(
                ApiError::bad("too many multipart fields", None).response(),
                &request_id,
            );
        }
        let name = field.name().unwrap_or("").to_owned();
        let filename = field.file_name().map(str::to_owned);
        let bytes = match tokio::time::timeout_at(read_deadline, field.bytes()).await {
            Ok(Ok(b)) => b,
            Err(_) => {
                state.metrics.timeout();
                return with_id(
                    ApiError {
                        status: StatusCode::GATEWAY_TIMEOUT,
                        message: "request body read timeout".into(),
                        param: None,
                    }
                    .response(),
                    &request_id,
                );
            }
            Ok(Err(e)) => {
                state.metrics.transcribe_error();
                let status = e.status();
                let mut error =
                    ApiError::bad(format!("Invalid multipart field: {}", e.body_text()), None);
                if status == StatusCode::PAYLOAD_TOO_LARGE {
                    error.status = status;
                    error.message = "request body too large".into();
                }
                return with_id(error.response(), &request_id);
            }
        };
        total += bytes.len();
        if name != "file" && bytes.len() > 16 * 1024 {
            state.metrics.transcribe_error();
            return with_id(
                ApiError::bad("multipart text field exceeds 16 KiB", None).response(),
                &request_id,
            );
        }
        parts.push((name, filename, bytes.to_vec()));
    }
    let input = match parse_fields(parts, state.context.capabilities.rnnt.is_some()) {
        Ok(v) => v,
        Err(e) => {
            state.metrics.transcribe_error();
            return with_id(e.response(), &request_id);
        }
    };
    if let Some(guard) = upload_guard.as_ref() {
        guard.release();
    }
    let admitted = match state.admission.admit_http() {
        Ok(v) => v,
        Err(crate::admission::AdmissionRefusal::Overloaded) => {
            state.metrics.overload();
            return with_id(
                ApiError {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    message: "service overloaded, retry later".into(),
                    param: None,
                }
                .response(),
                &request_id,
            );
        }
        Err(crate::admission::AdmissionRefusal::Draining) => {
            state.metrics.transcribe_error();
            return with_id(
                ApiError {
                    status: StatusCode::SERVICE_UNAVAILABLE,
                    message: "draining".into(),
                    param: None,
                }
                .response(),
                &request_id,
            );
        }
    };
    let control = gigaam_recognition::ExecutionControl::for_request();
    let owner = crate::admission::RequestWorkOwner::new(admitted, control.clone());
    let ctx = state.context.clone();
    let metrics = state.metrics.clone();
    let mut input = input;
    let bytes = std::mem::take(&mut input.audio);
    let q = OpenAiWorkInput {
        extension: input.extension.clone(),
        rnnt: input.engine == Engine::Rnnt,
    };
    let work = move |c: &gigaam_recognition::ExecutionControl| {
        http::do_openai(&ctx, &metrics, &q, bytes, c)
    };
    let out =
        http::execute_admitted_work(owner, control.clone(), state.request_timeout, work).await;
    let response = match out {
        http::AdmittedWorkResult::Completed(Ok(result)) => project(&result, &input),
        http::AdmittedWorkResult::Completed(Err(error)) => {
            state.metrics.transcribe_error();
            ApiError {
                status: error.status(),
                message: error.message(),
                param: None,
            }
            .response()
        }
        http::AdmittedWorkResult::TimedOut => {
            state.metrics.timeout();
            ApiError {
                status: StatusCode::GATEWAY_TIMEOUT,
                message: "processing timeout".into(),
                param: None,
            }
            .response()
        }
        http::AdmittedWorkResult::WorkerFailure => {
            state.metrics.transcribe_error();
            ApiError {
                status: StatusCode::SERVICE_UNAVAILABLE,
                message: "internal worker error".into(),
                param: None,
            }
            .response()
        }
    };
    let status = response.status().as_u16();
    state
        .metrics
        .observe_latency(started.elapsed().as_secs_f64());
    http::complete_admitted_response(
        response,
        &state.telemetry,
        &request_id,
        crate::log::TranscribeAccess {
            request_id: request_id.text().to_owned(),
            status,
            milliseconds: started.elapsed().as_millis(),
            bytes: total,
            model: match input.engine {
                Engine::Ctc => "ctc",
                Engine::Rnnt => "rnnt",
            }
            .to_owned(),
        },
    )
}

fn project(r: &TranscriptionResult, input: &Input) -> Response {
    let text = r
        .batch
        .channels()
        .first()
        .map(|c| gigaam_transcription::words_to_text(c.words()))
        .unwrap_or_default();
    match input.format {
        Format::Text => {
            ([(header::CONTENT_TYPE, "text/plain; charset=utf-8")], text).into_response()
        }
        Format::Json => (
            [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
            obj(vec![("text", Json::str(text))]).to_string(),
        )
            .into_response(),
        Format::Verbose => {
            let mut fields = vec![
                ("task", Json::str("transcribe")),
                ("duration", Json::round(r.duration_sec, 2)),
                ("text", Json::str(text)),
            ];
            if input.segment {
                fields.push((
                    "segments",
                    Json::Array(
                        r.batch
                            .segments()
                            .iter()
                            .enumerate()
                            .map(|(i, s)| {
                                obj(vec![
                                    ("id", Json::Usize(i)),
                                    ("start", Json::round(s.start(), 2)),
                                    ("end", Json::round(s.end(), 2)),
                                    ("text", Json::str(s.text())),
                                ])
                            })
                            .collect(),
                    ),
                ))
            }
            if input.word {
                fields.push((
                    "words",
                    Json::Array(
                        r.batch
                            .channels()
                            .iter()
                            .flat_map(|c| c.words())
                            .map(|w| {
                                obj(vec![
                                    ("word", Json::str(w.text())),
                                    ("start", Json::round(w.start(), 2)),
                                    ("end", Json::round(w.end(), 2)),
                                ])
                            })
                            .collect(),
                    ),
                ))
            }
            (
                [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
                obj(fields).to_string(),
            )
                .into_response()
        }
    }
}
fn with_id(mut r: Response, id: &RequestId) -> Response {
    r.headers_mut().insert("x-request-id", id.header().clone());
    r
}
async fn models(State(state): State<HttpState>) -> Response {
    let mut data = vec![model("gigaam-v3"), model("gigaam-v3-ctc")];
    if state.context.capabilities.rnnt.is_some() {
        data.push(model("gigaam-v3-rnnt"))
    }
    (
        [(header::CONTENT_TYPE, JSON_CONTENT_TYPE)],
        obj(vec![
            ("object", Json::str("list")),
            ("data", Json::Array(data)),
        ])
        .to_string(),
    )
        .into_response()
}
fn model(id: &str) -> Json {
    obj(vec![
        ("id", Json::str(id)),
        ("object", Json::str("model")),
        ("owned_by", Json::str("local")),
        ("created", Json::Int(0)),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;
    use tower::ServiceExt;
    fn wav_silence() -> Vec<u8> {
        let samples = vec![0u8; 320 * 2];
        let mut wav = Vec::new();
        wav.extend_from_slice(b"RIFF");
        wav.extend_from_slice(&(36u32 + samples.len() as u32).to_le_bytes());
        wav.extend_from_slice(b"WAVEfmt ");
        wav.extend_from_slice(&16u32.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&1u16.to_le_bytes());
        wav.extend_from_slice(&16_000u32.to_le_bytes());
        wav.extend_from_slice(&32_000u32.to_le_bytes());
        wav.extend_from_slice(&2u16.to_le_bytes());
        wav.extend_from_slice(&16u16.to_le_bytes());
        wav.extend_from_slice(b"data");
        wav.extend_from_slice(&(samples.len() as u32).to_le_bytes());
        wav.extend_from_slice(&samples);
        wav
    }
    fn multipart(fields: &[(&str, &str, &[u8])]) -> (String, Vec<u8>) {
        let boundary = "test-boundary";
        let mut body = Vec::new();
        for (name, filename, data) in fields {
            body.extend_from_slice(
                format!(
                    "--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"{}\r\n\r\n",
                    if filename.is_empty() {
                        String::new()
                    } else {
                        format!("; filename=\"{filename}\"")
                    }
                )
                .as_bytes(),
            );
            body.extend_from_slice(data);
            body.extend_from_slice(b"\r\n");
        }
        body.extend_from_slice(format!("--{boundary}--\r\n").as_bytes());
        (format!("multipart/form-data; boundary={boundary}"), body)
    }
    async fn test_router(
        max_http: usize,
        limit: usize,
        timeout: std::time::Duration,
    ) -> (
        axum::Router,
        crate::telemetry::test_support::PreparedHarness,
        std::sync::Arc<crate::admission::AdmissionState>,
    ) {
        let fixture = crate::tests::assembly_fixture().expect("test fixture builds");
        let harness =
            crate::telemetry::test_support::prepare(max_http, 8).expect("telemetry prepared");
        let metrics = harness.metrics_arc();
        let admission = std::sync::Arc::new(
            crate::admission::AdmissionState::new(
                &crate::ServiceAdmission::new(max_http, 8, timeout).unwrap(),
            )
            .unwrap(),
        );
        let mut state = crate::http::tests::state_with_telemetry(
            &fixture,
            admission.clone(),
            metrics,
            harness.producer(),
        )
        .unwrap();
        state.request_timeout = timeout;
        state.request_body_limit = crate::RequestBodyLimit::new(limit).unwrap();
        (crate::http::routes(state).into_router(), harness, admission)
    }
    async fn call(
        router: &axum::Router,
        content_type: &str,
        body: Vec<u8>,
        request_id: Option<&str>,
    ) -> axum::response::Response {
        use tower::ServiceExt;
        let mut builder = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header("content-type", content_type);
        if let Some(id) = request_id {
            builder = builder.header("x-request-id", id)
        }
        router
            .clone()
            .oneshot(builder.body(axum::body::Body::from(body)).unwrap())
            .await
            .unwrap()
    }

    fn fields(v: Vec<(&str, Vec<u8>)>) -> Vec<(String, Option<String>, Vec<u8>)> {
        v.into_iter()
            .map(|(n, b)| (n.to_owned(), None, b))
            .collect()
    }
    #[tokio::test]
    async fn multipart_upload_read_timeout_returns_504_and_releases_upload_slot() {
        let (router, _harness, admission) =
            test_router(1, 8192, std::time::Duration::from_millis(100)).await;
        use futures_util::StreamExt;
        let boundary = "test-boundary";
        let head = format!(
            "--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"clip.wav\"\r\n\r\n"
        );
        let stream = futures_util::stream::iter(vec![Ok::<_, std::io::Error>(
            axum::body::Bytes::from(head),
        )])
        .chain(futures_util::stream::pending());
        let request = axum::http::Request::builder()
            .method("POST")
            .uri("/v1/audio/transcriptions")
            .header(
                "content-type",
                format!("multipart/form-data; boundary={boundary}"),
            )
            .body(axum::body::Body::from_stream(stream))
            .unwrap();
        let first_router = router.clone();
        let first = tokio::spawn(async move { first_router.oneshot(request).await.unwrap() });
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        let audio = wav_silence();
        let (ct, body) = multipart(&[("file", "clip.wav", &audio), ("model", "", b"gigaam-v3")]);
        let second = call(&router, &ct, body, Some("concurrent-upload")).await;
        assert_eq!(second.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(second.headers()["x-request-id"], "concurrent-upload");
        let response = first.await.unwrap();
        assert_eq!(response.status(), StatusCode::GATEWAY_TIMEOUT);
        assert!(
            admission.admit_upload().is_ok(),
            "timed-out read releases capacity"
        );
    }

    #[tokio::test]
    async fn multipart_http_formats_models_and_errors() {
        let (router, _harness, _admission) =
            test_router(2, 8192, std::time::Duration::from_secs(2)).await;
        let audio = wav_silence();
        let (ct, body) = multipart(&[("file", "clip.wav", &audio), ("model", "", b"gigaam-v3")]);
        let response = call(&router, &ct, body, Some("openai-test-id")).await;
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(response.headers()["x-request-id"], "openai-test-id");
        let body = axum::body::to_bytes(response.into_body(), 8192)
            .await
            .unwrap();
        assert!(
            std::str::from_utf8(&body)
                .unwrap()
                .starts_with("{\"text\":")
        );
        for (format, expected) in [("text", "text/plain"), ("verbose_json", "application/json")] {
            let (ct, body) = multipart(&[
                ("file", "clip.wav", &audio),
                ("model", "", b"gigaam-v3"),
                ("response_format", "", format.as_bytes()),
            ]);
            let response = call(&router, &ct, body, None).await;
            assert_eq!(response.status(), StatusCode::OK);
            assert!(
                response.headers()["content-type"]
                    .to_str()
                    .unwrap()
                    .starts_with(expected)
            );
            if format == "verbose_json" {
                let json = axum::body::to_bytes(response.into_body(), 8192)
                    .await
                    .unwrap();
                let json = std::str::from_utf8(&json).unwrap();
                assert!(json.contains("\"task\":\"transcribe\""));
                assert!(json.contains("\"segments\":["));
                assert!(!json.contains("\"words\":"));
            } else {
                let text = axum::body::to_bytes(response.into_body(), 8192)
                    .await
                    .unwrap();
                assert!(!std::str::from_utf8(&text).unwrap().starts_with('{'));
            }
        }
        for values in [vec!["word"], vec!["segment", "word"]] {
            let mut owned: Vec<(String, String, Vec<u8>)> = vec![
                ("file".into(), "clip.wav".into(), audio.clone()),
                ("model".into(), String::new(), b"gigaam-v3".to_vec()),
                (
                    "response_format".into(),
                    String::new(),
                    b"verbose_json".to_vec(),
                ),
            ];
            let both = values.len() == 2;
            for value in &values {
                owned.push((
                    "timestamp_granularities[]".into(),
                    String::new(),
                    value.as_bytes().to_vec(),
                ));
            }
            let refs: Vec<(&str, &str, &[u8])> = owned
                .iter()
                .map(|(n, f, b)| (n.as_str(), f.as_str(), b.as_slice()))
                .collect();
            let (ct, body) = multipart(&refs);
            let response = call(&router, &ct, body, None).await;
            assert_eq!(response.status(), StatusCode::OK);
            let bytes = axum::body::to_bytes(response.into_body(), 8192)
                .await
                .unwrap();
            let json = std::str::from_utf8(&bytes).unwrap();
            assert!(json.contains("\"words\":["));
            assert_eq!(json.contains("\"segments\":["), both);
        }
        let response = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .uri("/v1/models")
                    .body(axum::body::Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let listed = axum::body::to_bytes(response.into_body(), 2048)
            .await
            .unwrap();
        let listed = std::str::from_utf8(&listed).unwrap();
        assert!(listed.contains("gigaam-v3"));
        assert!(listed.contains("gigaam-v3-ctc"));
        assert!(!listed.contains("gigaam-v3-rnnt"));
        assert!(listed.contains("\"created\":0"));
        let (ct, body) = multipart(&[("file", "x.wav", &audio), ("model", "", b"whisper-1")]);
        let response = call(&router, &ct, body, None).await;
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let err = axum::body::to_bytes(response.into_body(), 2048)
            .await
            .unwrap();
        assert!(
            std::str::from_utf8(&err)
                .unwrap()
                .contains("invalid_request_error")
        );
        let malformed = call(
            &router,
            "application/json",
            Vec::new(),
            Some("bad-content-type"),
        )
        .await;
        assert_eq!(malformed.status(), StatusCode::BAD_REQUEST);
        assert_eq!(malformed.headers()["x-request-id"], "bad-content-type");
        let invalid_id = call(&router, &ct, Vec::new(), Some("")).await;
        assert_eq!(invalid_id.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn multipart_http_enforces_body_limit_and_shared_admission() {
        let (router, _harness, _admission) =
            test_router(1, 256, std::time::Duration::from_secs(2)).await;
        let audio = wav_silence();
        let (ct, body) = multipart(&[("file", "clip.wav", &audio), ("model", "", b"gigaam-v3")]);
        let response = call(&router, &ct, body, None).await;
        assert_eq!(response.status(), StatusCode::PAYLOAD_TOO_LARGE);
        let (router, _harness, admission) =
            test_router(1, 8192, std::time::Duration::from_secs(2)).await;
        let _held = admission.admit_http().unwrap();
        let (ct, body) = multipart(&[("file", "clip.wav", &audio), ("model", "", b"gigaam-v3")]);
        let response = call(&router, &ct, body, None).await;
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        let envelope = axum::body::to_bytes(response.into_body(), 2048)
            .await
            .unwrap();
        assert!(
            std::str::from_utf8(&envelope)
                .unwrap()
                .contains("server_error")
        );
        let native = router
            .clone()
            .oneshot(
                axum::http::Request::builder()
                    .method("POST")
                    .uri("/v1/transcribe?ext=wav")
                    .body(axum::body::Body::from(audio))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(native.status(), StatusCode::SERVICE_UNAVAILABLE);
        let native_body = axum::body::to_bytes(native.into_body(), 2048)
            .await
            .unwrap();
        assert!(
            std::str::from_utf8(&native_body)
                .unwrap()
                .starts_with("{\"error\":")
        );
    }

    #[test]
    fn accepts_supported_subset_and_defaults() {
        let input = parse_fields(
            fields(vec![
                ("file", b"audio".to_vec()),
                ("model", b"gigaam-v3".to_vec()),
            ]),
            false,
        )
        .unwrap();
        assert_eq!(input.format, Format::Json);
        assert_eq!(input.engine, Engine::Ctc);
        let verbose = parse_fields(
            fields(vec![
                ("file", b"audio".to_vec()),
                ("model", b"gigaam-v3-ctc".to_vec()),
                ("response_format", b"verbose_json".to_vec()),
                ("timestamp_granularities[]", b"word".to_vec()),
            ]),
            false,
        )
        .unwrap();
        assert!(verbose.word);
        assert!(!verbose.segment);
    }
    #[test]
    fn refuses_unknown_models_fields_duplicates_and_unsupported_options() {
        for v in [
            vec![("file", b"x".to_vec()), ("model", b"whisper-1".to_vec())],
            vec![
                ("file", b"x".to_vec()),
                ("file", b"y".to_vec()),
                ("model", b"gigaam-v3".to_vec()),
            ],
            vec![
                ("file", b"x".to_vec()),
                ("model", b"gigaam-v3".to_vec()),
                ("stream", b"true".to_vec()),
            ],
            vec![
                ("file", b"x".to_vec()),
                ("model", b"gigaam-v3".to_vec()),
                ("junk", b"".to_vec()),
            ],
        ] {
            assert!(parse_fields(fields(v), false).is_err());
        }
        assert!(parse_model("gigaam-v3-rnnt", false).is_err());
        assert!(matches!(
            parse_model("gigaam-v3-rnnt", true),
            Ok(Engine::Rnnt)
        ));
    }
    #[test]
    fn accepts_zero_temperature_as_numeric_form() {
        for value in [b"0".as_slice(), b"0.0", b"0.00"] {
            assert!(
                parse_fields(
                    fields(vec![
                        ("file", b"x".to_vec()),
                        ("model", b"gigaam-v3".to_vec()),
                        ("temperature", value.to_vec())
                    ]),
                    false
                )
                .is_ok()
            );
        }
    }

    #[test]
    fn filename_uses_only_recognized_extension() {
        assert_eq!(extension(Some("../../clip.WAV")), Some("wav".into()));
        assert_eq!(extension(Some("clip.webm")), None);
    }
}
