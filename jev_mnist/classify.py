"""Ask Jev which digit an MNIST image shows."""

from dataclasses import dataclass
from io import BytesIO

import numpy as np
import PIL.Image
from typesafe_sdk import AsyncTypeSafeClient, Choice, Image

QUESTION = "digit"
INSTRUCTIONS = (
    "This image is a handwritten digit from the MNIST dataset "
    "(a white digit on a black background). Which digit (0-9) is it?"
)
DIGITS = [str(d) for d in range(10)]
# Image input is experimental in Jev 1.14; the API rejects images without this opt-in header.
IMAGES_OPT_IN_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
# The API rejects (never resizes) images whose sides fall outside this range, with "400 Invalid image dimensions".
MIN_IMAGE_SIDE = 280
MAX_IMAGE_SIDE = 1568
DEFAULT_UPSCALE = 10  # 28 * 10 = 280 pixels per side, the smallest accepted size.


def to_png(pixels: np.ndarray, upscale: int = DEFAULT_UPSCALE) -> bytes:
    """Encode a 28x28 grayscale array as PNG, enlarged with nearest-neighbor scaling."""
    image = PIL.Image.fromarray(pixels, mode="L")
    if upscale > 1:
        image = image.resize((image.width * upscale, image.height * upscale), PIL.Image.Resampling.NEAREST)
    buffer = BytesIO()
    image.save(buffer, format="png")
    return buffer.getvalue()


@dataclass(frozen=True)
class Prediction:
    digit: int
    confidence: float
    model: str
    input_tokens: int


async def classify(client: AsyncTypeSafeClient, png: bytes, model: str) -> Prediction:
    response = await client.system_one(
        state=Image(png),
        questions={
            QUESTION: Choice(instructions=INSTRUCTIONS, criteria={d: None for d in DIGITS}),
        },
        model=model,
    )
    answer = response.choices[QUESTION]
    return Prediction(
        digit=int(answer.choice),
        confidence=answer.confidence,
        model=response.model,
        input_tokens=response.usage.input_tokens,
    )
