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
                    ("type", Json::str("invalid_request_error")),
                    ("param", self.param.map(Json::str).unwrap_or(Json::Null)),
                    ("code", Json::Null),
                ]),
            )])
            .to_string(),
        )
            .into_response()
    }
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
    loop {
        let field = match multipart.next_field().await {
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
        let name = field.name().unwrap_or("").to_owned();
        let filename = field.file_name().map(str::to_owned);
        let bytes = match field.bytes().await {
            Ok(b) => b,
            Err(e) => {
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
        parts.push((name, filename, bytes.to_vec()));
    }
    let input = match parse_fields(parts, state.context.capabilities.rnnt.is_some()) {
        Ok(v) => v,
        Err(e) => {
            state.metrics.transcribe_error();
            return with_id(e.response(), &request_id);
        }
    };
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
        .channels
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
                        r.segments
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
                        r.channels
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
    fn fields(v: Vec<(&str, Vec<u8>)>) -> Vec<(String, Option<String>, Vec<u8>)> {
        v.into_iter()
            .map(|(n, b)| (n.to_owned(), None, b))
            .collect()
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
