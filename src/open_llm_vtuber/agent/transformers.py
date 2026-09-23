import re
from typing import AsyncIterator, Tuple, Callable, List, Union, Dict, Any
from functools import wraps
from .output_types import Actions, SentenceOutput, DisplayText
from ..utils.tts_preprocessor import tts_filter as filter_text
from ..live2d_model import Live2dModel
from ..config_manager import TTSPreprocessorConfig
from ..utils.sentence_divider import SentenceDivider
from ..utils.sentence_divider import SentenceWithTags, TagState
from loguru import logger
from ..privacy_logging import mapping_log_fields, text_log_fields


def sentence_divider(
    faster_first_response: bool = True,
    segment_method: str = "pysbd",
    valid_tags: List[str] = None,
):
    """
    Decorator that transforms token stream into sentences with tags

    Args:
        faster_first_response: bool - Whether to enable faster first response
        segment_method: str - Method for sentence segmentation
        valid_tags: List[str] - List of valid tags to process
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[str, Dict[str, Any]]]
        ],  # Expects str or dict
    ) -> Callable[
        ..., AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]
    ]:  # Yields SentenceWithTags or dict
        @wraps(func)
        async def wrapper(
                *args, **kwargs
        ) -> AsyncIterator[Union[SentenceWithTags, Dict[str, Any]]]:
            divider = SentenceDivider(
                faster_first_response=faster_first_response,
                segment_method=segment_method,
                valid_tags=valid_tags or [],
            )
            stream_from_func = func(*args, **kwargs)

            async for item in divider.process_stream(stream_from_func):
                if isinstance(item, SentenceWithTags):
                    logger.debug(
                        "sentence_divider yielding sentence: text={}, tag_count={}",
                        text_log_fields(item.text),
                        len(item.tags),
                    )
                elif isinstance(item, dict):
                    logger.debug(
                        "sentence_divider yielding dict: {}", mapping_log_fields(item)
                    )
                yield item

        return wrapper

    return decorator


def actions_extractor(live2d_model: Live2dModel):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            stream = func(*args, **kwargs)
            pending_display = ""
            pending_expression = None
            pending_tags = []

            async for item in stream:
                if isinstance(item, SentenceWithTags):
                    sentence = item
                    # Bracketed emotion labels are a renderer contract.  Keep
                    # registered labels for action extraction, but remove any
                    # provider-invented labels before display, TTS, channel
                    # delivery, and the final response assembled downstream.
                    sanitize = getattr(
                        live2d_model, "remove_unknown_emotion_tags", None
                    )
                    if callable(sanitize):
                        sentence = SentenceWithTags(
                            text=sanitize(sentence.text),
                            tags=sentence.tags,
                        )
                    actions = Actions()
                    if not any(
                        tag.state in [TagState.START, TagState.END]
                        for tag in sentence.tags
                    ):
                        expressions = live2d_model.extract_emotion(sentence.text)

                        # Emotion-only and parenthesized action fragments have no
                        # useful playback duration.  Keep their complete display
                        # text and attach it to the next audible semantic unit.
                        cleaned = sentence.text
                        for key in live2d_model.emo_map.keys():
                            cleaned = re.sub(
                                re.escape(f"[{key}]"),
                                "",
                                cleaned,
                                flags=re.IGNORECASE,
                            )
                        cleaned = re.sub(r"[（(][^）)]*[）)]", "", cleaned).strip()
                        has_spoken_content = any(char.isalnum() for char in cleaned)

                        if not has_spoken_content:
                            pending_display += sentence.text
                            pending_tags = sentence.tags
                            if expressions:
                                # With no speech between two labels, the label
                                # nearest to the eventual spoken text wins.
                                pending_expression = expressions[-1]
                            continue

                        if pending_display:
                            sentence = SentenceWithTags(
                                text=pending_display + sentence.text,
                                tags=sentence.tags,
                            )
                            pending_display = ""

                        if expressions:
                            # One playback unit has one effective emotion.  This
                            # keeps frontend visuals and TTS reference routing in
                            # agreement for sequences such as
                            # [surprise](action)[shy]spoken text.
                            actions.expressions = [expressions[-1]]
                            pending_expression = None
                        elif pending_expression is not None:
                            actions.expressions = [pending_expression]
                            pending_expression = None

                    yield sentence, actions
                elif isinstance(item, dict):
                    yield item
                else:
                    logger.warning(
                        f"actions_extractor received unexpected type: {type(item)}"
                    )

            if pending_display:
                actions = Actions()
                if pending_expression is not None:
                    actions.expressions = [pending_expression]
                yield SentenceWithTags(
                    text=pending_display,
                    tags=pending_tags,
                ), actions

        return wrapper
    return decorator


def display_processor():
    """
    Decorator that processes text for display, passing through dicts.
    """

    def decorator(
        func: Callable[
            ..., AsyncIterator[Union[Tuple[SentenceWithTags, Actions], Dict[str, Any]]]
        ],  # Input type hint
    ) -> Callable[
        ...,
        AsyncIterator[
            Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
        ],
    ]:  # Output type hint
        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[
            Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
        ]:  # Yield type hint
            stream = func(*args, **kwargs)

            async for item in stream:
                if (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], SentenceWithTags)
                ):
                    sentence, actions = item
                    text = sentence.text
                    # Handle think tag states
                    for tag in sentence.tags:
                        if tag.name == "think":
                            if tag.state == TagState.START:
                                text = "("
                            elif tag.state == TagState.END:
                                text = ")"

                    display = DisplayText(text=text)  # Simplified DisplayText creation
                    yield sentence, display, actions  # Yield the tuple
                elif isinstance(item, dict):
                    # Pass through dictionaries
                    yield item
                else:
                    logger.warning(
                        f"display_processor received unexpected type: {type(item)}"
                    )

        return wrapper

    return decorator


def tts_filter(
    tts_preprocessor_config: TTSPreprocessorConfig = None,
):
    """
    Decorator that filters text for TTS, passing through dicts.
    Skips TTS for think tag content.
    """

    def decorator(
        func: Callable[
            ...,
            AsyncIterator[
                Union[Tuple[SentenceWithTags, DisplayText, Actions], Dict[str, Any]]
            ],
        ],
    ) -> Callable[
        ..., AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]
    ]:
        config = tts_preprocessor_config or TTSPreprocessorConfig()

        @wraps(func)
        async def wrapper(
            *args, **kwargs
        ) -> AsyncIterator[Union[SentenceOutput, Dict[str, Any]]]:
            stream = func(*args, **kwargs)

            # 跨句追踪括号状态
            in_parenthesis = False

            async for item in stream:
                if (
                    isinstance(item, tuple)
                    and len(item) == 3
                    and isinstance(item[1], DisplayText)
                ):
                    sentence, display, actions = item
                    if any(tag.name == "think" for tag in sentence.tags):
                        tts = ""
                    else:
                        text = display.text

                        # 跨句括号过滤
                        result = []
                        for char in text:
                            if char in ('（', '('):
                                in_parenthesis = True
                            elif char in ('）', ')'):
                                in_parenthesis = False
                            elif not in_parenthesis:
                                result.append(char)
                        text = ''.join(result).strip()

                        tts = filter_text(
                            text=text,
                            remove_special_char=config.remove_special_char,
                            ignore_brackets=config.ignore_brackets,
                            ignore_parentheses=config.ignore_parentheses,
                            ignore_asterisks=config.ignore_asterisks,
                            ignore_angle_brackets=config.ignore_angle_brackets,
                        )

                    logger.debug(
                        "Prepared display and TTS: display={}, tts={}",
                        text_log_fields(display.text),
                        text_log_fields(tts),
                    )

                    yield SentenceOutput(
                        display_text=display,
                        tts_text=tts,
                        actions=actions,
                    )
                elif isinstance(item, dict):
                    yield item
                else:
                    logger.warning(f"tts_filter received unexpected type: {type(item)}")

        return wrapper

    return decorator

