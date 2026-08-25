// openai.KimiClient — Kimi's OpenAI-compatible coding endpoint with native
// preserved-thinking semantics.

class KimiClient {
    model: string,
    api_key: ai.Credential?,
    base_url: ai.Credential?,
    request_timeout_ms: int?,
    time_to_first_token_timeout_ms: int?,
    request_body: baml.json.json?,
    headers: map<string, string>?,
    query_params: map<string, string>?,
    max_completion_tokens: int?,

    function new(
        model: string = "kimi-for-coding",
        api_key: ai.Credential? = null,
        base_url: ai.Credential? = null,
        request_timeout_ms: int? = null,
        time_to_first_token_timeout_ms: int? = null,
        request_body: baml.json.json? = null,
        headers: map<string, string>? = null,
        query_params: map<string, string>? = null,
        max_completion_tokens: int? = null,
    ) -> KimiClient {
        KimiClient {
            model: model,
            api_key: api_key,
            base_url: base_url,
            request_timeout_ms: request_timeout_ms,
            time_to_first_token_timeout_ms: time_to_first_token_timeout_ms,
            request_body: request_body,
            headers: headers,
            query_params: query_params,
            max_completion_tokens: max_completion_tokens,
        }
    }

    function resolved_api_key(self) -> string? {
        ai.wire.resolve_credential(self.api_key, "KIMI_API_KEY")
    }

    function resolved_base_url(self) -> string {
        ai.wire.resolve_credential(self.base_url, null) ?? "https://api.kimi.com/coding/v1"
    }

    function compat(self) -> root.internal.ChatCompat {
        root.internal
            .ChatCompat
            .new(
                // Preserve the existing client id so durable journals created
                // by the former generic ChatClient remain replayable.
                provider = "openai-chat",
                base_url = self.resolved_base_url(),
                api_key_env = "KIMI_API_KEY",
                api_key_required = true,
                collapse_text_content = true,
                send_stream_options = true,
                max_tokens_key = "max_completion_tokens",
                replay_reasoning_content = true,
            )
    }

    function params(self, preview: bool = false) -> root.internal.ChatParams {
        let defaults: baml.json.json = {
            "thinking": {
                "type": "enabled",
                "keep": "all",
            },
        };
        root.internal
            .ChatParams
            .new(
                model = self.model,
                api_key = if (preview) {
                    null
                } else {
                    self.resolved_api_key()
                },
                max_completion_tokens = self.max_completion_tokens,
                headers = self.headers,
                query_params = self.query_params,
                request_body = ai.wire.merge_request_body(defaults, self.request_body),
            )
    }

    implements ai.Client {
        function id(self) -> string {
            `openai-chat/${self.model}`
        }

        function render(self, input: ai.ModelTurnInput) -> baml.http.Request {
            openai.internal.chat_preview(self.compat(), self.params(preview = true), input) catch_all (e) {
                _ => throw ai.errors.normalize(e),
            }
        }

        function invoke(self, input: ai.ModelTurnInput) -> ai.ModelTurn {
            root.internal.chat_invoke(
                self.compat(),
                self.params(),
                input,
                request_timeout_ms = self.request_timeout_ms,
            ) catch_all (e) {
                _ => throw ai.errors.normalize_invoke(e),
            }
        }
    }

    implements ai.stream.StreamingClient {
        function invoke_stream(self, input: ai.ModelTurnInput) -> ai.stream.TurnStream {
            root.internal.chat_invoke_stream(
                self.compat(),
                self.params(),
                input,
                request_timeout_ms = self.request_timeout_ms,
                time_to_first_token_timeout_ms = self.time_to_first_token_timeout_ms,
            ) catch_all (e) {
                _ => throw ai.errors.normalize_invoke(e),
            }
        }
    }
}
