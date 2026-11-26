import os
import re
from collections import deque
from contextlib import contextmanager
from http import HTTPStatus
from threading import Lock
from time import monotonic, sleep


DEFAULT_GEMINI_MODEL = 'gemini-3.5-flash-lite'
PUBLISHED_FREE_TIER_MODELS = (
    DEFAULT_GEMINI_MODEL,
    'gemini-4-hydrogen',
    'gemini-3.8-flash',
)
MODEL_ID_PATTERN = re.compile(r'gemini-[a-z0-9][a-z0-9.-]{0,119}')
RETRYABLE_STATUS_CODES = {429, 500, 503, 504}


class GeminiRequestError(RuntimeError):
    """A credential-safe error suitable for the command line."""

    def __init__(self, message, *, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class GenerationRateLimiter:
    """Pace generation requests across the ETL and modelling stages."""

    def __init__(self, requests_per_minute=15):
        if requests_per_minute < 1:
            raise ValueError('The generation rate limit must be positive.')
        self.requests_per_minute = requests_per_minute
        self._starts = deque()
        self._lock = Lock()

    def wait_for_request_slot(self):
        while True:
            with self._lock:
                now = monotonic()
                while self._starts and now - self._starts[0] >= 60:
                    self._starts.popleft()
                if len(self._starts) < self.requests_per_minute:
                    self._starts.append(now)
                    return
                wait = 60 - (now - self._starts[0])
            sleep(wait + 0.25)


def require_gemini_key():
    key = os.environ.get('GEMINI_API_KEY', '').strip()
    if not key or key == 'YOUR_API_KEY':
        raise GeminiRequestError('Enter your Gemini API key using the masked PowerShell prompt.')
    return key


def _public_model_id(value, key):
    """Accept only Gemini model IDs that are safe to display."""
    if not isinstance(value, str):
        return None
    model = value.strip().removeprefix('models/')
    if MODEL_ID_PATTERN.fullmatch(model) and key not in model:
        return model
    return None


def resolve_gemini_model(model=None):
    """Prefer the command-line model, then the environment, then the default."""
    value = model if model is not None else os.environ.get('GEMINI_MODEL', '').strip()
    if model is None and not value:
        value = DEFAULT_GEMINI_MODEL
    selected = _public_model_id(value, require_gemini_key())
    if selected is None:
        raise GeminiRequestError('Choose a Gemini model ID using --list-gemini-models.')
    return selected


@contextmanager
def _gemini_connection(key, *, operation='request'):
    try:
        import httpx
        from google import genai
        from google.genai import errors, types
    except ImportError:
        raise GeminiRequestError('Install google-genai in the active Python environment.') from None
    except OSError:
        raise GeminiRequestError(
            'The Google SDK could not initialise its network configuration.'
        ) from None

    http_options = types.HttpOptions(retry_options=types.HttpRetryOptions(attempts=1))
    try:
        with genai.Client(api_key=key, vertexai=False, http_options=http_options) as client:
            yield client
    except errors.APIError as error:
        code = error.code
        if type(code) is int and 100 <= code <= 599:
            try:
                status = f'{code} {HTTPStatus(code).phrase}'
            except ValueError:
                status = str(code)
            message = f'Gemini HTTP {status} during {operation}.'
            if code == 503:
                # Only display structured API errors. A gateway may return a raw HTML body.
                if error.status == 'UNAVAILABLE' and isinstance(error.message, str):
                    detail = ' '.join(redact_gemini_text(error.message).split())
                    message += f' UNAVAILABLE: {detail[:500]}'
                else:
                    message += ' No structured UNAVAILABLE message was provided.'
            raise GeminiRequestError(
                message, status_code=code,
            ) from None
        raise GeminiRequestError('Gemini returned an API error. Check Google AI Studio.') from None
    except httpx.TimeoutException:
        raise GeminiRequestError('The connection to Gemini timed out.') from None
    except (httpx.RequestError, OSError):
        raise GeminiRequestError('Could not reach Gemini. Check your network connection.') from None
    except Exception:
        raise GeminiRequestError('The Gemini SDK could not complete the request.') from None


def list_gemini_models():
    """Fetch the Gemini generation catalogue without sending an inference prompt."""
    key = require_gemini_key()
    models = set()
    with _gemini_connection(key, operation='model listing') as client:
        for entry in client.models.list(config={'page_size': 1000}):
            model = _public_model_id(entry.name, key)
            if model and 'generateContent' in (entry.supported_actions or ()):
                models.add(model)
    return [model for model in PUBLISHED_FREE_TIER_MODELS if model in models] + sorted(
        models.difference(PUBLISHED_FREE_TIER_MODELS),
    )


def redact_gemini_text(text):
    """Remove credential values before sending feedback or writing a log."""
    for name in ('GEMINI_API_KEY', 'GOOGLE_API_KEY'):
        key = os.environ.get(name, '').strip()
        if key:
            text = text.replace(key, '[redacted]')
    return re.sub(r'AIza[A-Za-z0-9_-]{35}|(?<![\w-])AQ[A-Za-z0-9_-]{26,}', '[redacted]', text)


def upload_csvs(paths, *, model):
    key = require_gemini_key()
    with _gemini_connection(key, operation=f'CSV upload for {model}') as client:
        return tuple(
            client.files.upload(file=str(path), config={'mime_type': 'text/csv', 'display_name': path.name})
            for path in paths
        )


def generate_candidates(prompt, files, *, model, temperature, count=1, retained=None, rate_limiter=None):
    """Generate candidates, keeping replies in retained when a later call fails."""
    key = require_gemini_key()
    supports_batch = model.startswith(('gemini-2.0-', 'gemini-2.5-'))
    config = {}
    if supports_batch or model.startswith(('gemini-3.0-', 'gemini-3.1-', 'gemini-3.5-')):
        config['temperature'] = temperature
    if supports_batch and count > 1:
        config['candidate_count'] = count

    candidates = [] if retained is None else retained
    with _gemini_connection(key, operation=f'generation with {model}') as client:
        for _ in range(1 if supports_batch else count - len(candidates)):
            if not supports_batch and count > 1:
                print(f'Requesting candidate {len(candidates) + 1} of {count}.', flush=True)
            if rate_limiter is not None:
                rate_limiter.wait_for_request_slot()
            response = client.models.generate_content(
                model=model, contents=[redact_gemini_text(prompt), *files], config=config or None,
            )
            version = _public_model_id(response.model_version, key)
            if version and version != model:
                print(f'Gemini reported model version {version}.', flush=True)
            texts = [
                redact_gemini_text('\n'.join(
                    part.text for part in candidate.content.parts or () if part.text and not part.thought
                ).strip())
                for candidate in response.candidates or () if candidate.content
            ]
            if supports_batch:
                candidates.extend(texts)
            else:
                candidates.append(texts[0] if texts else '')
    return tuple(candidates) if candidates else ('',)
