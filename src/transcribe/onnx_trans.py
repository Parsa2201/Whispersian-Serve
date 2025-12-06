# import sys
# import time
from collections import namedtuple
import platform
# import queue
import zlib
from logging import getLogger

import numpy as np

# import original modules
# sys.path.append("./util")
from src.transcribe.onnx_trans_utils.decode_utils import (
    ApplyTimestampRules,
    BeamSearchDecoder,
    GreedyDecoder,
    MaximumLikelihoodRanker,
    SuppressBlank,
    SuppressTokens,
)
from src.transcribe.onnx_trans_utils.math_utils import softmax
# from util.microphone_utils import start_microphone_input  # noqa
from src.transcribe.onnx_trans_utils.model_utils import check_and_download_models, check_and_download_file  # noqa
from src.transcribe.onnx_trans_utils.languages import LANGUAGES #, TO_LANGUAGE_CODE
# from util.arg_utils import get_base_parser, get_savepath, update_parser  # noqa

from src.transcribe.onnx_trans_utils.audio_utils import (
    CHUNK_LENGTH,
    HOP_LENGTH,
    N_FRAMES,
    N_SAMPLES,
    SAMPLE_RATE,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
)

disable_ailia_tokenizer=False
if not disable_ailia_tokenizer:
    from src.transcribe.onnx_trans_utils.ailia_tokenizer import get_tokenizer
else:
    from src.transcribe.onnx_trans_utils.tokenizer import get_tokenizer

DecodingResult = namedtuple(
    "DecodingResult",
    [
        "audio_features",
        "language",
        "language_probs",
        "tokens",
        "text",
        "avg_logprob",
        "no_speech_prob",
        "temperature",
    ],
)
class OnnxTrans:

    def __init__(self, 
                 model_path=None,
                #  encoder_weight_path=None,
                #  decoder_weight_path=None,
                #  encoder_model_path=None,
                #  decoder_model_path=None,
                 model_type="base", 
                 language="fa",
                 task="transcribe",
                 fp16=False,
                 temperature=0,
                 temperature_increment_on_fallback=0.2,
                 compression_ratio_threshold=2.4,
                 logprob_threshold=-1.0,
                 no_speech_threshold=0.6,
                 best_of=5,
                 beam_size=None,
                 patience=None,
                 length_penalty=None,
                 suppress_tokens="-1"):
        if model_path is None:
            encoder_weight_path = None
            decoder_weight_path = None
            encoder_model_path = None
            decoder_model_path = None
        else:
            encoder_weight_path = model_path + "/encoder.onnx"
            decoder_weight_path = model_path + "/decoder.onnx"
            encoder_model_path = model_path + "/encoder.onnx.prototxt"
            decoder_model_path = model_path + "/decoder.onnx.prototxt"
        self.model_type = model_type
        self.fp16 = fp16
        self.language = language
        self.task = task
        self.temperature = temperature
        self.temperature_increment_on_fallback = temperature_increment_on_fallback
        self.compression_ratio_threshold = compression_ratio_threshold
        self.logprob_threshold = logprob_threshold
        self.no_speech_threshold = no_speech_threshold
        self.best_of = best_of
        self.beam_size = beam_size
        self.patience = patience
        self.length_penalty = length_penalty
        self.suppress_tokens = suppress_tokens

        self.logger = getLogger(__name__)

        ModelDimensions = namedtuple(
            "ModelDimensions",
            [
                "n_mels",
                "n_audio_ctx",
                "n_audio_state",
                "n_audio_head",
                "n_audio_layer",
                "n_vocab",
                "n_text_ctx",
                "n_text_state",
                "n_text_head",
                "n_text_layer",
            ],
        )

        dims_dict = {
            "tiny": ModelDimensions(80, 1500, 384, 6, 4, 51865, 448, 384, 6, 4),
            "base": ModelDimensions(80, 1500, 512, 8, 6, 51865, 448, 512, 8, 6),
            "small": ModelDimensions(80, 1500, 768, 12, 12, 51865, 448, 768, 12, 12),
            "medium": ModelDimensions(80, 1500, 1024, 16, 24, 51865, 448, 1024, 16, 24),
            "large": ModelDimensions(80, 1500, 1280, 20, 32, 51865, 448, 1280, 20, 32),
            "large-v3": ModelDimensions(128, 1500, 1280, 20, 32, 51866, 448, 1280, 20, 32),
            "turbo": ModelDimensions(128, 1500, 1280, 20, 32, 51866, 448, 1280, 20, 4),
        }
        self.dims = dims_dict[model_type]

        self.LAYER_NORM_ENABLE = False
        if fp16:
            self.LAYER_NORM_ENABLE = True

        self.OPT = ".opt"
        self.OPT2 = ".opt2"
        self.OPT3 = ".opt"

        self.FP16 = ""
        if fp16:
            self.FP16 = "_fp16"


        self.WEIGHT_DEC_TINY_PATH = decoder_weight_path or "decoder_tiny_fix_kv_cache" +  self.FP16 + self.OPT2 + ".onnx"
        self.WEIGHT_DEC_BASE_PATH = decoder_weight_path or "decoder_base_fix_kv_cache" + self.FP16 + self.OPT2 + ".onnx"
        self.WEIGHT_DEC_SMALL_PATH = decoder_weight_path or "decoder_small_fix_kv_cache" + self.FP16 + self.OPT2 + ".onnx"
        self.WEIGHT_DEC_MEDIUM_PATH = decoder_weight_path or "decoder_medium_fix_kv_cache" + self.FP16 + self.OPT2 + ".onnx"
        self.WEIGHT_DEC_LARGE_PATH = decoder_weight_path or "decoder_large_fix_kv_cache.onnx"
        self.WEIGHT_DEC_LARGE_V3_PATH = decoder_weight_path or "decoder_large_v3_fix_kv_cache.onnx"
        self.WEIGHT_DEC_TURBO_PATH = decoder_weight_path or "decoder_turbo_fix_kv_cache" + self.FP16 + self.OPT3 + ".onnx"

        self.MODEL_DEC_TINY_PATH = decoder_model_path or "decoder_tiny_fix_kv_cache" +  self.FP16 + self.OPT2 + ".onnx.prototxt"
        self.MODEL_DEC_BASE_PATH = decoder_model_path or "decoder_base_fix_kv_cache" + self.FP16 + self.OPT2 + ".onnx.prototxt"
        self.MODEL_DEC_SMALL_PATH = decoder_model_path or "decoder_small_fix_kv_cache" + self.FP16 + self.OPT2 + ".onnx.prototxt"
        self.MODEL_DEC_MEDIUM_PATH = decoder_model_path or "decoder_medium_fix_kv_cache" + self.FP16 + self.OPT2 + ".onnx.prototxt"
        self.MODEL_DEC_LARGE_PATH = decoder_model_path or "decoder_large_fix_kv_cache.onnx.prototxt"
        self.MODEL_DEC_LARGE_V3_PATH = decoder_model_path or "decoder_large_v3_fix_kv_cache.onnx.prototxt"
        self.MODEL_DEC_TURBO_PATH = decoder_model_path or "decoder_turbo_fix_kv_cache" + self.FP16 + self.OPT3 + ".onnx.prototxt"

        self.WEIGHT_ENC_TINY_PATH = encoder_weight_path or "encoder_tiny" + self.FP16 + self.OPT + ".onnx"
        self.WEIGHT_ENC_BASE_PATH = encoder_weight_path or "encoder_base" + self.FP16 + self.OPT + ".onnx"
        self.WEIGHT_ENC_SMALL_PATH = encoder_weight_path or "encoder_small" + self.FP16 + self.OPT + ".onnx"
        self.WEIGHT_ENC_MEDIUM_PATH = encoder_weight_path or "encoder_medium" + self.FP16 + self.OPT + ".onnx"
        self.WEIGHT_ENC_LARGE_PATH = encoder_weight_path or "encoder_large.onnx"
        self.WEIGHT_ENC_LARGE_V3_PATH = encoder_weight_path or "encoder_large_v3.onnx"
        self.WEIGHT_ENC_TURBO_PATH = encoder_weight_path or "encoder_turbo" + self.FP16 + self.OPT3 + ".onnx"

        self.MODEL_ENC_TINY_PATH = encoder_model_path or "encoder_tiny" + self.FP16 + self.OPT + ".onnx.prototxt"
        self.MODEL_ENC_BASE_PATH = encoder_model_path or "encoder_base" + self.FP16 + self.OPT + ".onnx.prototxt"
        self.MODEL_ENC_SMALL_PATH = encoder_model_path or "encoder_small" + self.FP16 + self.OPT + ".onnx.prototxt"
        self.MODEL_ENC_MEDIUM_PATH = encoder_model_path or "encoder_medium" + self.FP16 + self.OPT + ".onnx.prototxt"
        self.MODEL_ENC_LARGE_PATH = encoder_model_path or "encoder_large.onnx.prototxt"
        self.MODEL_ENC_LARGE_V3_PATH = encoder_model_path or "encoder_large_v3.onnx.prototxt"
        self.MODEL_ENC_TURBO_PATH = encoder_model_path or "encoder_turbo"  + self.FP16 + self.OPT3 + ".onnx.prototxt"

        self.WEIGTH_ENC_LARGE_PB_PATH = "encoder_large_weights.pb"
        self.WEIGHT_DEC_LARGE_PB_PATH = "decoder_large_weights.pb"
        self.WEIGHT_DEC_LARGE_FIX_KV_CACHE_PB_PATH = "decoder_large_fix_kv_cache_weights.pb"
        self.WEIGTH_ENC_LARGE_V3_PB_PATH = "encoder_large_v3_weights.pb"
        self.WEIGHT_DEC_LARGE_V3_PB_PATH = "decoder_large_v3_weights.pb"
        self.WEIGHT_DEC_LARGE_V3_FIX_KV_CACHE_PB_PATH = "decoder_large_v3_fix_kv_cache_weights.pb"
        self.WEIGHT_ENC_TURBO_PB_PATH = "encoder_turbo_weights" + self.OPT3 + ".pb"

        self.REMOTE_PATH = "https://storage.googleapis.com/ailia-models/whisper/"

    def is_multilingual(self):
        return self.dims.n_vocab >= 51865


    def num_languages(self):
        return self.dims.n_vocab - 51765 - int(self.is_multilingual())


    def format_timestamp(self, seconds: float, always_include_hours: bool = False):
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours}:" if always_include_hours or hours > 0 else ""

        return f"{hours_marker}{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


    def get_initial_tokens(self, tokenizer, options):
        sot_sequence = tokenizer.sot_sequence
        sample_len = options.get("sample_len") or self.dims.n_text_ctx // 2
        n_ctx = self.dims.n_text_ctx

        tokens = list(sot_sequence)
        prefix = options.get("prefix", None)
        prompt = options.get("prompt", [])

        if prefix:
            prefix_tokens = (
                tokenizer.encode(" " + prefix.strip())
                if isinstance(prefix, str)
                else prefix
            )
            if sample_len is not None:
                max_prefix_len = n_ctx // 2 - sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        # Probably don't need this
        # if prompt or args.prompt:
        #     if args.prompt:
        #         prompt_arg_tokens = tokenizer.encode(args.prompt)
        #         prompt_tokens = prompt
        #         prev_prompt_len = (n_ctx // 2 - 1) - len(prompt_arg_tokens)
        #         tokens = (
        #             [tokenizer.sot_prev]
        #             + prompt_arg_tokens
        #             + prompt_tokens[-prev_prompt_len:]
        #             + tokens
        #         )
        #     else:
        #         prompt_tokens = (
        #             tokenizer.encode(" " + prompt.strip())
        #             if isinstance(prompt, str)
        #             else prompt
        #         )
        #         tokens = [tokenizer.sot_prev] + prompt_tokens[-(n_ctx // 2 - 1) :] + tokens

        return tuple(tokens)


    def get_suppress_tokens(self, tokenizer, options):
        suppress_tokens = options["suppress_tokens"]

        if isinstance(suppress_tokens, str):
            suppress_tokens = [int(t) for t in suppress_tokens.split(",")]

        if -1 in suppress_tokens:
            suppress_tokens = [t for t in suppress_tokens if t >= 0]
            suppress_tokens.extend(tokenizer.non_speech_tokens)
        elif suppress_tokens is None or len(suppress_tokens) == 0:
            suppress_tokens = []  # interpret empty string as an empty list
        else:
            assert isinstance(suppress_tokens, list), "suppress_tokens must be a list"

        suppress_tokens.extend([tokenizer.sot, tokenizer.sot_prev, tokenizer.sot_lm])
        if tokenizer.no_speech is not None:
            # no-speech probability is collected separately
            suppress_tokens.append(tokenizer.no_speech)

        return tuple(sorted(set(suppress_tokens)))


    def new_kv_cache(self, n_group, length=451):
        model_type = self.model_type
        if model_type == "tiny.en" or model_type == "tiny":
            size = [8, n_group, length, 384]
        elif model_type == "base.en" or model_type == "base":
            size = [12, n_group, length, 512]
        elif model_type == "small.en" or model_type == "small":
            size = [24, n_group, length, 768]
        elif model_type == "medium.en" or model_type == "medium":
            size = [48, n_group, length, 1024]
        elif model_type in ("large", "large-v3"):
            size = [64, n_group, length, 1280]
        elif model_type == "turbo":
            size = [8, n_group, length, 1280]
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        return np.zeros(size, dtype=np.float32, order="C")


    def compression_ratio(self, text) -> float:
        return len(text) / len(zlib.compress(text.encode("utf-8")))
    
    def get_audio_features(self, enc_net, mel):
        # if args.benchmark:
        #     start = int(round(time.time() * 1000))

        mel = mel.astype(np.float32)
        output = enc_net.run(None, {"mel": mel})
        audio_features = output[0]

        # if args.benchmark:
        #     end = int(round(time.time() * 1000))
        #     estimation_time = end - start
        #     logger.info(f"\tencoder processing time {estimation_time} ms")

        return audio_features
    
    def inference_logits(
        self,
        dec_net,
        tokens,
        audio_features,
        kv_cache=None,
        initial_token_length=None,
        constant_audio_feature=False,
    ):
        n_group = tokens.shape[0]
        initial_token_length = (
            initial_token_length if initial_token_length else tokens.shape[-1]
        )
        is_init_kv_cache = False
        if kv_cache is None:
            # if not self.dynamic_kv_cache:
            kv_cache = self.new_kv_cache(n_group)
            # else:
            #     kv_cache = self.new_kv_cache(n_group, initial_token_length)
            offset = 0
            length = initial_token_length
            is_init_kv_cache = True
        else:
            offset = kv_cache.shape[2]
            # if not self.dynamic_kv_cache:
            length = offset + 1
            _kv_cache = self.new_kv_cache(n_group)
            _kv_cache[:, :, :offset, :] = kv_cache
            # else:
            #     _kv_cache = self.new_kv_cache(n_group, offset + 1)
            #     _kv_cache[:, :, :-1, :] = kv_cache
            kv_cache = _kv_cache

        if tokens.shape[-1] > initial_token_length:
            # only need to use the last token except in the first forward pass
            tokens = tokens[:, -1:]

        tokens = tokens.astype(np.int64)
        offset = np.array(offset, dtype=np.int64)

        # if args.benchmark:
        #     start = int(round(time.time() * 1000))


        kv_cache = kv_cache.astype(np.float32)
        output = dec_net.run(
            None,
            {
                "tokens": tokens,
                "audio_features": audio_features,
                "kv_cache": kv_cache,
                "offset": offset,
            },
        )
        logits, kv_cache = output

        # if args.benchmark:
        #     end = int(round(time.time() * 1000))
        #     estimation_time = end - start
        #     logger.info(f"\tdecoder processing time {estimation_time} ms")

        # if not self.dynamic_kv_cache:
        return logits, kv_cache[:, :, :length, :]
        # else:
        #     return logits, kv_cache
        
    def decode(self, enc_net, dec_net, mel, options):
        single = mel.ndim == 2
        if single:
            mel = mel.unsqueeze(0)

        language = options.get("language") or "en"
        tokenizer = get_tokenizer(
            self.is_multilingual(),
            num_languages=self.num_languages(),
            language=language,
            task=self.task,
        )

        n_group = options.get("beam_size") or options.get("best_of") or 1
        n_ctx = self.dims.n_text_ctx
        sample_len = options.get("sample_len") or self.dims.n_text_ctx // 2

        initial_tokens = self.get_initial_tokens(tokenizer, options)
        sample_begin = len(initial_tokens)
        sot_index = initial_tokens.index(tokenizer.sot)

        # logit filters: applies various rules to suppress or penalize certain tokens
        logit_filters = []
        if options.get("suppress_blank"):
            logit_filters.append(SuppressBlank(tokenizer, sample_begin))
        if options.get("suppress_tokens"):
            logit_filters.append(SuppressTokens(self.get_suppress_tokens(tokenizer, options)))
        if not options.get("without_timestamps"):
            precision = CHUNK_LENGTH / self.dims.n_audio_ctx  # usually 0.02 seconds
            max_initial_timestamp_index = None
            max_initial_timestamp = options.get("max_initial_timestamp")
            if max_initial_timestamp:
                max_initial_timestamp_index = round(max_initial_timestamp / precision)
            logit_filters.append(
                ApplyTimestampRules(tokenizer, sample_begin, max_initial_timestamp_index)
            )

        # sequence ranker: implements how to rank a group of sampled sequences
        sequence_ranker = MaximumLikelihoodRanker(options.get("length_penalty"))

        # decoder: implements how to select the next tokens, given the autoregressive distribution
        if options.get("beam_size") is not None:
            decoder = BeamSearchDecoder(
                options.get("beam_size"), tokenizer.eot, options.get("patience")
            )
        else:
            decoder = GreedyDecoder(options.get("temperature"), tokenizer.eot)

        decoder.reset()
        n_audio = mel.shape[0]

        audio_features = self.get_audio_features(enc_net, mel)
        tokens = np.repeat(np.array([initial_tokens]), n_audio, axis=-1)
        languages = [language] * audio_features.shape[0]

        # repeat the audio & text tensors by the group size, for beam search or best-of-n sampling
        audio_features = np.repeat(audio_features, n_group, axis=0)
        tokens = np.repeat(tokens, n_group, axis=0)

        n_batch = tokens.shape[0]
        sum_logprobs = np.zeros(n_batch)
        no_speech_probs = [np.nan] * n_batch
        initial_token_length = len(initial_tokens)
        kv_cache = None

        # sampling loop
        for i in range(sample_len):
            # if args.debug:
            #     start = int(round(time.time() * 1000))
            constant_audio_feature = i >= 2
            logits, kv_cache = self.inference_logits(
                dec_net,
                tokens,
                audio_features,
                kv_cache,
                initial_token_length,
                constant_audio_feature,
            )
            # if args.debug:
            #     end = int(round(time.time() * 1000))
            #     estimation_time = end - start
            #     logger.info(f"step: {i} / {sample_len} {estimation_time} ms")

            if i == 0 and tokenizer.no_speech is not None:  # save no_speech_probs
                probs_at_sot = softmax(logits[:, sot_index], axis=-1)
                no_speech_probs = probs_at_sot[:, tokenizer.no_speech].tolist()

            # now we need to consider the logits at the last token only
            logits = logits[:, -1]

            # apply the logit filters, e.g. for suppressing or applying penalty to
            for logit_filter in logit_filters:
                logit_filter.apply(logits, tokens)

            def rearrange_kv_cache(source_indices):
                kv_cache[...] = kv_cache[:, source_indices]

            # expand the tokens tensor with the selected next tokens
            tokens, completed = decoder.update(
                tokens, logits, sum_logprobs, rearrange_kv_cache
            )

            if completed or tokens.shape[-1] > n_ctx:
                break

            # if args.intermediate:
            #     texts = [tokenizer.decode(t[len(initial_tokens) :]).strip() for t in tokens]
            #     print(texts[0][-32:] + "\n\u001B[2A")

        # reshape the tensors to have (n_audio, n_group) as the first two dimensions
        audio_features = audio_features[::n_group]
        no_speech_probs = no_speech_probs[::n_group]
        assert audio_features.shape[0] == len(no_speech_probs) == n_audio

        tokens = tokens.reshape(n_audio, n_group, -1)
        sum_logprobs = sum_logprobs.reshape(n_audio, n_group)

        # get the final candidates for each group, and slice between the first sampled token and EOT
        tokens, sum_logprobs = decoder.finalize(tokens, sum_logprobs)
        tokens = [
            [t[sample_begin : np.nonzero(t == tokenizer.eot)[0][0]] for t in s]
            for s in tokens
        ]

        # select the top-ranked sample in each group
        selected = sequence_ranker.rank(tokens, sum_logprobs)
        tokens = [t[i].tolist() for i, t in zip(selected, tokens)]
        texts = [tokenizer.decode(t).strip() for t in tokens]

        sum_logprobs = [lp[i] for i, lp in zip(selected, sum_logprobs)]
        avg_logprobs = [lp / (len(t) + 1) for t, lp in zip(tokens, sum_logprobs)]

        fields = (texts, languages, tokens, audio_features, avg_logprobs, no_speech_probs)
        if len(set(map(len, fields))) != 1:
            raise RuntimeError(f"inconsistent result lengths: {list(map(len, fields))}")

        result = [
            DecodingResult(
                audio_features=features,
                language=language,
                language_probs=None,
                tokens=tokens,
                text=text,
                avg_logprob=avg_logprob,
                no_speech_prob=no_speech_prob,
                temperature=options.get("temperature"),
            )
            for text, language, tokens, features, avg_logprob, no_speech_prob in zip(
                *fields
            )
        ]

        if single:
            result = result[0]

        return result

    def decode_with_fallback(self, enc_net, dec_net, segment, decode_options):
        logprob_threshold = decode_options.get("logprob_threshold", -1.0)
        temperature = decode_options.get("temperature", 0)
        no_speech_threshold = decode_options.get("no_speech_threshold", 0.6)
        compression_ratio_threshold = decode_options.get("compression_ratio_threshold", 2.4)

        temperatures = (
            [temperature] if isinstance(temperature, (int, float)) else temperature
        )
        decode_result = None

        for t in temperatures:
            kwargs = {**decode_options}
            if t > 0:
                # disable beam_size and patience when t > 0
                kwargs.pop("beam_size", None)
                kwargs.pop("patience", None)
                print("temperature", t)
            else:
                # disable best_of when t == 0
                kwargs.pop("best_of", None)

            options = {**kwargs, "temperature": t}
            decode_result = self.decode(enc_net, dec_net, segment, options)[0]

            needs_fallback = False
            if (
                compression_ratio_threshold is not None
                and self.compression_ratio(decode_result.text) > compression_ratio_threshold
            ):
                needs_fallback = True  # too repetitive
            if (
                logprob_threshold is not None
                and decode_result.avg_logprob < logprob_threshold
            ):
                needs_fallback = True  # average log probability is too low
            if (
                no_speech_threshold is not None
                and decode_result.no_speech_prob > no_speech_threshold
            ):
                needs_fallback = False  # silence
            if not needs_fallback:
                break

        return [decode_result]

    def predict(self, wav, enc_net, dec_net, immediate=False):
        language = self.language
        temperature = self.temperature
        temperature_increment_on_fallback = self.temperature_increment_on_fallback
        compression_ratio_threshold = self.compression_ratio_threshold
        logprob_threshold = self.logprob_threshold
        no_speech_threshold = self.no_speech_threshold

        if temperature_increment_on_fallback is not None:
            temperature = tuple(
                np.arange(temperature, 1.0 + 1e-6, temperature_increment_on_fallback)
            )
        else:
            temperature = [temperature]

        decode_options = {
            "task": self.task,
            "language": language,
            "temperature": temperature,
            "best_of": self.best_of,
            "beam_size": self.beam_size,
            "patience": self.patience,
            "length_penalty": self.length_penalty,
            "suppress_tokens": self.suppress_tokens,
            "compression_ratio_threshold": compression_ratio_threshold,
            "logprob_threshold": logprob_threshold,
            "no_speech_threshold": self.no_speech_threshold,
            "suppress_blank": True,
            "prompt": [],
        }

        mel = log_mel_spectrogram(wav, self.dims.n_mels, padding=N_SAMPLES)
        content_frames = mel.shape[-1] - N_FRAMES

        if language is None:
            segment = pad_or_trim(mel, N_FRAMES)
            _, probs = self.detect_language(enc_net, dec_net, segment)
            decode_options["language"] = language = max(probs, key=probs.get)
            self.logger.info(
                f"Detected language: {LANGUAGES[decode_options['language']].title()}"
            )

        mel = np.expand_dims(mel, axis=0)
        task = decode_options.get("task", self.task)
        tokenizer = get_tokenizer(
            self.is_multilingual(), num_languages=self.num_languages(), language=language, task=task
        )

        seek = 0
        input_stride = N_FRAMES // self.dims.n_audio_ctx  # mel frames per output token: 2
        time_precision = (
            input_stride * HOP_LENGTH / SAMPLE_RATE
        )  # time per output token: 0.02 (seconds)
        all_tokens = []
        all_segments = []
        prompt_reset_since = 0

        def new_segment(*, start: float, end: float, tokens, result: DecodingResult):
            tokens = tokens.tolist()
            text_tokens = [token for token in tokens if token < tokenizer.eot]
            return {
                "seek": seek,
                "start": start,
                "end": end,
                "text": tokenizer.decode(text_tokens),
                "tokens": tokens,
                "temperature": result.temperature,
                "avg_logprob": result.avg_logprob,
                "no_speech_prob": result.no_speech_prob,
            }

        # try:
        #     import tqdm


        #     pbar = tqdm.tqdm(
        #         total=content_frames, unit="frames", disable=immediate is not False
        #     )
        # except ImportError:
        #     pbar = None

        # show the progress bar when verbose is False (otherwise the transcribed text will be printed)
        while seek < content_frames:
            time_offset = float(seek * HOP_LENGTH / SAMPLE_RATE)
            mel_segment = mel[:, :, seek : seek + N_FRAMES]
            segment_size = min(N_FRAMES, content_frames - seek)
            segment_duration = segment_size * HOP_LENGTH / SAMPLE_RATE
            mel_segment = pad_or_trim(mel_segment, N_FRAMES)

            decode_options["prompt"] = all_tokens[prompt_reset_since:]
            result = self.decode_with_fallback(enc_net, dec_net, mel_segment, decode_options)
            result = result[0]
            tokens = np.array(result.tokens)

            if no_speech_threshold is not None:
                # no voice activity check
                should_skip = result.no_speech_prob > no_speech_threshold
                if logprob_threshold is not None and result.avg_logprob > logprob_threshold:
                    # don't skip if the logprob is high enough, despite the no_speech_prob
                    should_skip = False

                if should_skip:
                    seek += segment_size  # fast-forward to the next segment boundary
                    continue

            previous_seek = seek
            current_segments = []

            timestamp_tokens = tokens >= tokenizer.timestamp_begin
            single_timestamp_ending = timestamp_tokens[-2:].tolist() == [False, True]

            consecutive = np.where(timestamp_tokens[:-1] & timestamp_tokens[1:])[0] + 1
            if len(consecutive) > 0:
                # if the output contains two consecutive timestamp tokens
                slices = consecutive.tolist()
                if single_timestamp_ending:
                    slices.append(len(tokens))

                last_slice = 0
                for current_slice in slices:
                    sliced_tokens = tokens[last_slice:current_slice]
                    start_timestamp_pos = (
                        sliced_tokens[0].item() - tokenizer.timestamp_begin
                    )
                    end_timestamp_pos = sliced_tokens[-1].item() - tokenizer.timestamp_begin
                    current_segments.append(
                        new_segment(
                            start=time_offset + start_timestamp_pos * time_precision,
                            end=time_offset + end_timestamp_pos * time_precision,
                            tokens=sliced_tokens,
                            result=result,
                        )
                    )
                    last_slice = current_slice

                if single_timestamp_ending:
                    # single timestamp at the end means no speech after the last timestamp.
                    seek += segment_size
                else:
                    # otherwise, ignore the unfinished segment and seek to the last timestamp
                    last_timestamp_pos = (
                        tokens[last_slice - 1].item() - tokenizer.timestamp_begin
                    )
                    seek += last_timestamp_pos * input_stride
            else:
                duration = segment_duration
                timestamps = tokens[np.ravel(timestamp_tokens.nonzero())]
                if (
                    len(timestamps) > 0
                    and timestamps[-1].item() != tokenizer.timestamp_begin
                ):
                    # no consecutive timestamps but it has a timestamp; use the last one.
                    last_timestamp_pos = timestamps[-1].item() - tokenizer.timestamp_begin
                    duration = last_timestamp_pos * time_precision

                current_segments.append(
                    new_segment(
                        start=time_offset,
                        end=time_offset + duration,
                        tokens=tokens,
                        result=result,
                    )
                )
                seek += segment_size

            if immediate:
                for segment in current_segments:
                    start, end, text = segment["start"], segment["end"], segment["text"]
                    line = f"[{self.format_timestamp(start)} --> {self.format_timestamp(end)}] {text}"
                    print(line)

            # if a segment is instantaneous or does not contain text, clear it
            for i, segment in enumerate(current_segments):
                if segment["start"] == segment["end"] or segment["text"].strip() == "":
                    segment["text"] = ""
                    segment["tokens"] = []
                    segment["words"] = []

            all_segments.extend(
                [
                    {"id": i, **segment}
                    for i, segment in enumerate(current_segments, start=len(all_segments))
                ]
            )
            all_tokens.extend(
                [token for segment in current_segments for token in segment["tokens"]]
            )

            if result.temperature > 0.5:
                # do not feed the prompt tokens if a high temperature was used
                prompt_reset_since = len(all_tokens)

            # if pbar is not None:
            #     # update progress bar
            #     pbar.update(min(content_frames, seek) - previous_seek)

        d = dict(
            text=tokenizer.decode(all_tokens), segments=all_segments, language=language
        )
        return d

    def recognize_from_audio(self, inputs, enc_net, dec_net):
        immediate = False

        outputs = []
        # input audio loop
        for audio_path in inputs:
            self.logger.info(audio_path)

            # prepare input data
            wav = load_audio(audio_path)

            # inference
            self.logger.info("Start inference...")
            # if args.benchmark:
            #     logger.info("BENCHMARK mode")
            #     total_time_estimation = 0
            #     start = int(round(time.time() * 1000))
            #     output = predict(
            #         wav, enc_net, dec_net, immediate=immediate, microphone=False
            #     )
            #     end = int(round(time.time() * 1000))
            #     estimation_time = end - start
            #     logger.info(f"\ttotal processing time {estimation_time} ms")
            # else:
            output = self.predict(wav, enc_net, dec_net, immediate=immediate)
            outputs.append(output)

            if not immediate:
                # output result
                for res in output["segments"]:
                    self.logger.info(
                        f"[{self.format_timestamp(res['start'])} --> {self.format_timestamp(res['end'])}] {res['text']}"
                    )

        self.logger.info("Script finished successfully.")

        return outputs

    def transcribe(self, inputs):
        # global WEIGHT_DEC_PATH, MODEL_DEC_PATH, WEIGHT_ENC_PATH, MODEL_ENC_PATH
        model_dic = {
            "tiny": {
                "enc": (self.WEIGHT_ENC_TINY_PATH, self.MODEL_ENC_TINY_PATH),
                "dec": (self.WEIGHT_DEC_TINY_PATH, self.MODEL_DEC_TINY_PATH),
            },
            "base": {
                "enc": (self.WEIGHT_ENC_BASE_PATH, self.MODEL_ENC_BASE_PATH),
                "dec": (self.WEIGHT_DEC_BASE_PATH, self.MODEL_DEC_BASE_PATH),
            },
            "small": {
                "enc": (self.WEIGHT_ENC_SMALL_PATH, self.MODEL_ENC_SMALL_PATH),
                "dec": (self.WEIGHT_DEC_SMALL_PATH, self.MODEL_DEC_SMALL_PATH),
            },
            "medium": {
                "enc": (self.WEIGHT_ENC_MEDIUM_PATH, self.MODEL_ENC_MEDIUM_PATH),
                "dec": (self.WEIGHT_DEC_MEDIUM_PATH, self.MODEL_DEC_MEDIUM_PATH),
            },
            "large": {
                "enc": (self.WEIGHT_ENC_LARGE_PATH, self.MODEL_ENC_LARGE_PATH),
                "dec": (self.WEIGHT_DEC_LARGE_PATH, self.MODEL_DEC_LARGE_PATH),
            },
            "large-v3": {
                "enc": (self.WEIGHT_ENC_LARGE_V3_PATH, self.MODEL_ENC_LARGE_V3_PATH),
                "dec": (self.WEIGHT_DEC_LARGE_V3_PATH, self.MODEL_DEC_LARGE_V3_PATH),
            },
            "turbo": {
                "enc": (self.WEIGHT_ENC_TURBO_PATH, self.MODEL_ENC_TURBO_PATH),
                "dec": (self.WEIGHT_DEC_TURBO_PATH, self.MODEL_DEC_TURBO_PATH),
            },
        }
        model_info = model_dic[self.model_type]

        self.WEIGHT_ENC_PATH, self.MODEL_ENC_PATH = model_info["enc"]
        self.WEIGHT_DEC_PATH, self.MODEL_DEC_PATH = model_info["dec"]
        check_and_download_models(self.WEIGHT_ENC_PATH, self.MODEL_ENC_PATH, self.REMOTE_PATH)
        check_and_download_models(self.WEIGHT_DEC_PATH, self.MODEL_DEC_PATH, self.REMOTE_PATH)
        if self.model_type == "large":
            check_and_download_file(self.WEIGTH_ENC_LARGE_PB_PATH, self.REMOTE_PATH)
            # if self.dynamic_kv_cache:
            #     check_and_download_file(self.WEIGHT_DEC_LARGE_PB_PATH, self.REMOTE_PATH)
            # else:
            check_and_download_file(self.WEIGHT_DEC_LARGE_FIX_KV_CACHE_PB_PATH, self.REMOTE_PATH)
        elif self.model_type == "large-v3":
            check_and_download_file(self.WEIGTH_ENC_LARGE_V3_PB_PATH, self.REMOTE_PATH)
            # if self.dynamic_kv_cache:
            #     check_and_download_file(self.WEIGHT_DEC_LARGE_V3_PB_PATH, self.REMOTE_PATH)
            # else:
            check_and_download_file(
                self.WEIGHT_DEC_LARGE_V3_FIX_KV_CACHE_PB_PATH, self.REMOTE_PATH
            )
        elif self.model_type == "turbo":
            if self.fp16 == False:
                check_and_download_file(self.WEIGHT_ENC_TURBO_PB_PATH, self.REMOTE_PATH)

        pf = platform.system()
        if pf == "Darwin":
            self.logger.info(
                "This model not optimized for macOS GPU currently."
                " So we will use BLAS (env_id = 1)."
            )
            self.env_id = 1
        else:
            self.logger.info(
                "This model uses a lot of memory."
                " If an error occurs during execution, specify -e 0 and execute on the CPU."
            )

        # initialize
        import onnxruntime

        providers = ["CPUExecutionProvider"]
        # providers = ["CUDAExecutionProvider"]
        enc_net = onnxruntime.InferenceSession(self.WEIGHT_ENC_PATH, providers=providers)
        # if args.profile:
        #     options = onnxruntime.SessionOptions()
        #     options.enable_profiling = True
        #     dec_net = onnxruntime.InferenceSession(
        #         WEIGHT_DEC_PATH, options, providers=providers
        #     )
        # else:
        dec_net = onnxruntime.InferenceSession(self.WEIGHT_DEC_PATH, providers=providers)


        return [_['text'] for _ in self.recognize_from_audio(inputs, enc_net, dec_net)]

        # if args.profile:
        #     if args.onnx:
        #         prof_file = dec_net.end_profiling()
        #         print(prof_file)
        #     else:
        #         print(dec_net.get_summary())

if __name__ == "__main__":
    f = OnnxTrans(
                # decoder_weight_path='./export_model/decoder.opt.onnx',
                #   decoder_model_path='./export_model/decoder.opt.onnx.prototxt',
                #   encoder_weight_path='./export_model/encoder.opt.onnx',
                #   encoder_model_path='./export_model/encoder.opt.onnx.prototxt'
                  model_path='./export_model'
                  )
    output = f.transcribe(["/Users/parsa/Downloads/Varzesh_BaharVarzesh_79/audios_chunked_79/Varzesh_BaharVarzesh0_2.wav",
                           "/Users/parsa/Downloads/Varzesh_BaharVarzesh_79/audios_chunked_79/Varzesh_BaharVarzesh0_8.wav"
                           ])
    print(output)