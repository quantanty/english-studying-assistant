import logging
from langchain_core.utils.pydantic import is_basemodel_subclass
from collections.abc import Callable, Mapping
from difflib import get_close_matches
from typing import Any, Dict, List, Literal, Optional, Sequence, Union, cast

from operator import itemgetter

import proto
from google import genai
from google.api_core.exceptions import (
    FailedPrecondition,
    GoogleAPIError,
    InvalidArgument,
    ResourceExhausted,
    ServiceUnavailable,
)
from google.genai.client import Client
from google.genai.errors import ClientError, ServerError
from google.genai.types import (
    Candidate,
    Content,
    GenerateContentConfig,
    GenerateContentResponse,
    GenerationConfig,
    HttpOptions,
    Part,
    ThinkingConfig,
)
from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
    is_data_content_block,
)
from langchain_core.messages.ai import UsageMetadata, add_usage, subtract_usage
from langchain_core.output_parsers import (
    JsonOutputParser,
    PydanticOutputParser,
)
from langchain_core.output_parsers.base import OutputParserLike
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    ChatResult,
)
from langchain_core.runnables import Runnable, RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_core.utils import secret_from_env
# from langchain_google_genai.chat_models import _convert_to_parts, _parse_chat_history
from pydantic import BaseModel, Field, SecretStr, model_validator
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)
from typing_extensions import Self

logger = logging.getLogger(__name__)

_allowed_params_prediction_service_genai = [
    "model",
    "contents",
    "config",
]

def _append_to_content(
    current_content: str | list[Any] | None, new_item: Any
) -> str | list[Any]:
    """Appends a new item to the content, handling different initial content types."""
    if current_content is None and isinstance(new_item, str):
        return new_item
    if current_content is None:
        return [new_item]
    if isinstance(current_content, str):
        return [current_content, new_item]
    if isinstance(current_content, list):
        current_content.append(new_item)
        return current_content
    # This case should ideally not be reached with proper type checking,
    # but it catches any unexpected types that might slip through.
    msg = f"Unexpected content type: {type(current_content)}"
    raise TypeError(msg)

def _parse_response_candidate(
    response_candidate: Candidate,
    model_name: str | None = None,
) -> AIMessage:
    content: None | str | list[str | dict] = None
    additional_kwargs: dict[str, Any] = {}
    response_metadata: dict[str, Any] = {"model_provider": "google_genai"}
    if model_name:
        response_metadata["model_name"] = model_name
    
    parts = response_candidate.content.parts or [] if response_candidate.content else []
    for part in parts:
        text: str | None = None
        try:
            if hasattr(part, "text") and part.text is not None:
                text = part.text
        except AttributeError:
            pass
        if text is not None and text:
            content = _append_to_content(content, text)

    return AIMessage(
        content=content,
        additional_kwargs=additional_kwargs,
        response_metadata=response_metadata
    )

def _response_to_result(
    response: GenerateContentResponse,
    stream: bool = False,
    prev_usage: UsageMetadata | None = None,
) -> ChatResult:
    llm_output = (
        {"prompt_feedback": response.prompt_feedback.model_dump()}
        if response.prompt_feedback
        else {}
    )

    # Get usage metadata
    try:
        if response.usage_metadata is None:
            msg = "Usage metadata is None"
            raise AttributeError(msg)
        input_tokens = response.usage_metadata.prompt_token_count or 0
        thought_tokens = response.usage_metadata.thoughts_token_count or 0
        output_tokens = (
            response.usage_metadata.candidates_token_count or 0
        ) + thought_tokens
        total_tokens = response.usage_metadata.total_token_count or 0
        cache_read_tokens = response.usage_metadata.cached_content_token_count or 0
        if input_tokens + output_tokens + cache_read_tokens + total_tokens > 0:
            if thought_tokens > 0:
                cumulative_usage = UsageMetadata(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_token_details={"cache_read": cache_read_tokens},
                    output_token_details={"reasoning": thought_tokens},
                )
            else:
                cumulative_usage = UsageMetadata(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    total_tokens=total_tokens,
                    input_token_details={"cache_read": cache_read_tokens},
                )
            # previous usage metadata needs to be subtracted because gemini api returns
            # already-accumulated token counts with each chunk
            lc_usage = subtract_usage(cumulative_usage, prev_usage)
            if prev_usage and cumulative_usage["input_tokens"] < prev_usage.get(
                "input_tokens", 0
            ):
                # Gemini 2.0 returns a lower cumulative count of prompt tokens
                # in the final chunk. We take this count to be ground truth because
                # it's consistent with the reported total tokens. So we need to
                # ensure this chunk compensates (the subtract_usage funcction floors
                # at zero).
                lc_usage["input_tokens"] = cumulative_usage[
                    "input_tokens"
                ] - prev_usage.get("input_tokens", 0)
        else:
            lc_usage = None
    except AttributeError:
        lc_usage = None

    generations: list[ChatGeneration] = []

    for candidate in response.candidates or []:
        generation_info: dict[str, Any] = {}
        if candidate.finish_reason:
            generation_info["finish_reason"] = candidate.finish_reason.name
            # Add model_name in last chunk
            generation_info["model_name"] = response.model_version or ""
        generation_info["safety_ratings"] = (
            [safety_rating.model_dump() for safety_rating in candidate.safety_ratings]
            if candidate.safety_ratings
            else []
        )
        message = _parse_response_candidate(candidate)

        if not hasattr(message, "response_metadata"):
            message.response_metadata = {}

        try:
            if candidate.grounding_metadata:
                grounding_metadata = candidate.grounding_metadata.model_dump()
                # Ensure None fields that are expected to be lists become empty lists
                # to prevent errors in downstream processing
                if (
                    "grounding_supports" in grounding_metadata
                    and grounding_metadata["grounding_supports"] is None
                ):
                    grounding_metadata["grounding_supports"] = []
                if (
                    "grounding_chunks" in grounding_metadata
                    and grounding_metadata["grounding_chunks"] is None
                ):
                    grounding_metadata["grounding_chunks"] = []
                if (
                    "web_search_queries" in grounding_metadata
                    and grounding_metadata["web_search_queries"] is None
                ):
                    grounding_metadata["web_search_queries"] = []
                generation_info["grounding_metadata"] = grounding_metadata
                message.response_metadata["grounding_metadata"] = grounding_metadata
        except AttributeError:
            pass

        message.usage_metadata = lc_usage

        if stream:
            generations.append(
                ChatGenerationChunk(
                    message=cast("AIMessageChunk", message),
                    generation_info=generation_info,
                )
            )
        else:
            generations.append(
                ChatGeneration(message=message, generation_info=generation_info)
            )
    if not response.candidates:
        # Likely a "prompt feedback" violation (e.g., toxic input)
        # Raising an error would be different than how OpenAI handles it,
        # so we'll just log a warning and continue with an empty message.
        logger.warning(
            "Gemini produced an empty response. Continuing with empty message\n"
            f"Feedback: {response.prompt_feedback}"
        )
        if stream:
            generations = [
                ChatGenerationChunk(
                    message=AIMessageChunk(
                        content="",
                        response_metadata={
                            "prompt_feedback": proto.Message.to_dict(
                                response.prompt_feedback
                            )
                        },
                    ),
                    generation_info={},
                )
            ]
        else:
            generations = [ChatGeneration(message=AIMessage(""), generation_info={})]
    return ChatResult(generations=generations, llm_output=llm_output)

def _convert_to_parts(
    raw_content: str | Sequence[str | dict],
) -> list[Part]:
    content = [raw_content] if isinstance(raw_content, str) else raw_content

    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(Part(text=part))
        elif isinstance(part, Mapping):
            if "type" in part:
                if part["type"] == "text":
                    ...
    return parts

def _parse_chat_history(
    input_messages: Sequence[BaseMessage]
) -> tuple[Content | None, List[Content]]:
    input_messages = list(input_messages)
    
    formatted_messages: list[Content] = []
    
    system_instruction: Content | None = None
    messages_without_tool_messages = [
        message for message in input_messages if not isinstance(message, ToolMessage)
    ]
    tool_messages = [
        message for message in input_messages if isinstance(message, ToolMessage)
    ]
    for i, message in enumerate(messages_without_tool_messages):
        if isinstance(message, SystemMessage):
            system_parts = _convert_to_parts(message.content)
            if i == 0:
                system_instruction = Content(parts=system_parts)
            elif system_instruction is not None:
                if system_instruction.parts is None:
                    system_instruction.parts = system_parts
                else:
                    system_instruction.parts.extend(system_parts)
            else:
                pass
            continue
        if isinstance(message, AIMessage):
            role = "model"
            parts = _convert_to_parts(message.content)
        elif isinstance(message, HumanMessage):
            role = "user"
            parts = _convert_to_parts(message.content)
        else:
            msg = f"Unexpeted message with type {type(message)} at the position {i}."
            raise ValueError(msg)
            
        formatted_messages.append(Content(role=role, parts=parts))
    return system_instruction, formatted_messages



class _BaseGoogleGenAI(BaseModel):
    model: str = Field(...)
    
    google_api_key: SecretStr | None = Field(
        alias="api_key",
        default_factory=secret_from_env(
            ["GOOGLE_API_KEY", "GEMINI_API_KEY"], default=None
        )
    )

    credentials: Any = None
    
    temperature: float = 0.7
    
    top_p: float | None = None
    
    top_k: int | None = None
    
    max_output_tokens: int | None = Field(default=None, alias="max_tokens")

    n: int = 1
    
    max_retries: int = Field(default=6, alias="retries")
    
    timeout: float | None = Field(default=None, alias="request_timeout")
    
    client_options: dict | None = Field(
        default=None
    )

    base_url: str | None = Field(
        default=None
    )

    transport: str | None = Field(
        default=None,
        alias="api_transport",
    )
    
    additional_headers: dict[str, str] | None = Field(
        default=None,
    )
    
    # response_modalities: list[Modality] | None = Field(
    #     default=None,
    # )
    
    thinking_budget: int | None = Field(
        default=None,
    )

    include_thoughts: bool | None = Field(
        default=None,
    )
    

class ChatGoogleGenAIJSON(_BaseGoogleGenAI, BaseChatModel):
    """
    A REST-based Gemini ChatModel for LangChain
    - Supports JSON Schema fully (anyOf, oneOf, allOf, $defs)
    - Supports tool calling
    - Supports JSON Mode via response_schema
    """

    thinking_level: Literal["low", "high"] | None = Field(
        default=None,
    )
    
    client: Any = Field(default=None, exclude=True)
    
    default_metadata: Sequence[tuple[str, str]] | None = Field(
        default=None,
        alias="default_metadata_input",
    )
   
    response_mime_type: str | None = None
    
    response_schema: dict[str, Any] | None = None
    
    stop: list[str] | None = None
    
    streaming: bool | None = None
    
    model_kwargs: dict[str, Any] = Field(default_factory=dict)

    def __init__(self, **kwargs):
        valid_fields = set()
        for field_name, field_info in self.__class__.model_fields.items():
            valid_fields.add(field_name)
            if hasattr(field_info, "alias") and field_info.alias is not None:
                valid_fields.add(field_info.alias)
                
        for arg in kwargs:
            if arg not in valid_fields:
                suggestions = get_close_matches(arg, valid_fields, n=1)
                suggestion = (
                    f" Did you mean: '{suggestions[0]}'?" if suggestions else ""
                )
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_environment(self) -> Self:
        """Validates params and builds client."""
        if self.temperature is not None and not 0 <= self.temperature <= 2.0:
            msg = "temperature must be in the range [0.0, 2.0]"
            raise ValueError(msg)

        if self.top_p is not None and not 0 <= self.top_p <= 1:
            msg = "top_p must be in the range [0.0, 1.0]"
            raise ValueError(msg)

        if self.top_k is not None and self.top_k <= 0:
            msg = "top_k must be positive"
            raise ValueError(msg)

        additional_headers = self.additional_headers or {}
        self.default_metadata = tuple(additional_headers.items())

        google_api_key = None
        if not self.credentials:
            if isinstance(self.google_api_key, SecretStr):
                google_api_key = self.google_api_key.get_secret_value()
            else:
                google_api_key = self.google_api_key

        base_url = self.base_url
        if isinstance(self.base_url, dict):
            # Handle case where base_url is provided as a dict
            # (Backwards compatibility for deprecated client_options field)
            if keys := list(self.base_url.keys()):
                if "api_endpoint" in keys and len(keys) == 1:
                    base_url = self.base_url["api_endpoint"]
                elif "api_endpoint" in keys and len(keys) > 1:
                    msg = (
                        "When providing base_url as a dict, it can only contain the "
                        "api_endpoint key. Extra keys found: "
                        f"{[k for k in keys if k != 'api_endpoint']}"
                    )
                    raise ValueError(msg)
                else:
                    msg = (
                        "When providing base_url as a dict, it must only contain the "
                        "api_endpoint key."
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    "base_url must be a string or a dict containing the "
                    "api_endpoint key."
                )
                raise ValueError(msg)

        http_options = HttpOptions(
            base_url=cast("str", base_url), headers=additional_headers
        )

        if google_api_key:
            self.client = Client(api_key=google_api_key, http_options=http_options)
        else:
            project_id = getattr(self.credentials, "project_id", None)
            location = getattr(self.credentials, "location", "us-central1")
            self.client = Client(
                vertexai=True,
                project=project_id,
                location=location,
                http_options=http_options,
            )
        return self


    def _build_base_generation_config(
        self, stop: list[str] | None, **kwargs: Any
    ) -> dict[str, Any]:
        """Build the base generation configuration from instance attributes."""
        config: dict[str, Any] = {
            "candidate_count": self.n,
            "temperature": self.temperature,
            "stop_sequences": stop,
            "max_output_tokens": self.max_output_tokens,
            "top_k": self.top_k,
            "top_p": self.top_p,
            # "response_modalities": self.response_modalities,
        }
        return {k: v for k, v in config.items() if v is not None}

    def _merge_generation_config(
        self, base_config: dict[str, Any], generation_config: dict[str, Any]
    ) -> dict[str, Any]:
        processed_config = dict(generation_config)
        return {**base_config, **processed_config}

    def _validate_and_add_response_schema(
        self, gen_config: dict[str, Any],
        response_schema: dict[str, Any],
        response_mime_type: str | None,
    ) -> None:
        if response_mime_type != "application/json":
            error_message = (
                "JSON schema structured output is only supported when "
                "response_mime_type is set to 'application/json'"
            )
            if response_mime_type == "text/x.enum":
                error_message += (
                    ". Instead of 'text/x.enum', define enums using your JSON schema."
                )
            raise ValueError(error_message)

        gen_config["response_json_schema"] = response_schema
    
    def _add_response_parameters(
        self, gen_config: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        response_mime_type = kwargs.get("response_mime_type", self.response_mime_type)
        if response_mime_type is not None:
            gen_config["response_mime_type"] = response_mime_type
            
        response_schema = kwargs.get("response_schema", self.response_schema)
        response_json_schema = kwargs.get("response_json_schema")
        
        schema_to_use = (
            response_json_schema
            if response_json_schema is not None
            else response_schema
        )
        if schema_to_use:
            self._validate_and_add_response_schema(
                gen_config=gen_config,
                response_schema=schema_to_use,
                response_mime_type=response_mime_type,
            )

        return gen_config
    
    def _prepare_params(
        self,
        stop: list[str] | None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> GenerationConfig:
        gen_config = self._build_base_generation_config(stop, **kwargs)
        if generation_config:
            gen_config = self._merge_generation_config(gen_config, generation_config)
            
        gen_config = self._add_response_parameters(gen_config, **kwargs)
        return GenerationConfig.model_validate(gen_config)
    
    def _build_request_config(
        self,
        params: GenerationConfig,
        system_instruction: Content | None,
        stop: list[str] | None,
        **kwargs: Any,
    ) -> GenerateContentConfig:
        thinking_config = None
        if params.thinking_config is not None:
            thinking_config = ThinkingConfig(
                include_thoughts=params.thinking_config.include_thoughts,
                thinking_budget=params.thinking_config.thinking_budget,
            )

        return GenerateContentConfig(
            thinking_config=thinking_config,
            system_instruction=system_instruction,
            stop_sequences=stop,
            **kwargs,
        )
    
    def _prepare_request(
        self,
        messages: list[BaseMessage],
        *,
        stop: list[str] | None = None,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        system_instruction, history = _parse_chat_history(
            messages
        )
        
        params = self._prepare_params(
            stop, generation_config=generation_config, **kwargs
        )

        request = self._build_request_config(
            params,
            system_instruction,
            stop,
            **kwargs,
        )

        return {"model": self.model, "contents": history, "config": request}
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager = None,
        *,
        generation_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> ChatResult:

        if self.client is None:
            msg = "Client not initialized."
            raise ValueError(msg)
        
        request = self._prepare_request(
            messages,
            stop=stop,
            generation_config=generation_config,
            **kwargs,
        )
        response = self.client.models.generate_content(**request)
        return _response_to_result(response)

    def with_structured_output(
        self,
        schema: dict | type[BaseModel],
        method: Literal["function_calling", "json_mode", "json_schema"]
        | None = "json_schema",
        *,
        include_raw: bool = False,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, dict | BaseModel]:
        _ = kwargs.pop("strict", None)
        if kwargs:
            msg = f"Received unsupported arguments {kwargs}"
            raise ValueError(msg)

        parser: OutputParserLike

        if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
            msg = f"Unsupported schema type {type(schema)}"
            raise ValueError(msg)
        schema_json = schema.model_json_schema()
        
        llm = self.bind(
            response_mime_type="application/json",
            response_json_schema=schema_json
        )
        parser = PydanticOutputParser(pydantic_object=cast(type[BaseModel], schema))

        if include_raw:
            parser_with_fallback = RunnablePassthrough.assign(
                parsed=itemgetter("raw") | parser, parsing_error=lambda _: None
            ).with_fallbacks(
                [RunnablePassthrough.assign(parsed=lambda _: None)],
                exception_key="parsing_error",
            )
            return {"raw": llm} | parser_with_fallback
        
        return llm | parser

    # ------------------------------------------------------
    # Required by LangChain (for hierarchy and serialization)
    # ------------------------------------------------------
    @property
    def _llm_type(self) -> str:
        return "google-genai-json-rest"

    def get_token_ids(self, text: str) -> List[int]:
        # Not strictly needed; implement naive fallback
        return [ord(c) for c in text]
