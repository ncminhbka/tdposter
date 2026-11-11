import os
from pathlib import Path

import torch
import torch.nn as nn
from einops import rearrange
from PIL import Image
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Mistral3ForConditionalGeneration,
    pipeline,
)

from .sampling import cap_pixels, concatenate_images
from .system_messages import (
    PROMPT_IMAGE_INTEGRITY,
    PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
    PROMPT_TEXT_INTEGRITY,
    SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_UPSAMPLING_I2I,
    SYSTEM_MESSAGE_UPSAMPLING_T2I,
    SYSTEM_PROMPT_CONTENT_FILTER,
)

OUTPUT_LAYERS_MISTRAL = [10, 20, 30]
OUTPUT_LAYERS_QWEN3 = [9, 18, 27]
MAX_LENGTH = 512
NSFW_THRESHOLD = 0.85
UPSAMPLING_MAX_IMAGE_SIZE = 768**2


class Mistral3SmallEmbedder(nn.Module):
    def __init__(
        self,
        model_spec: str = "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
        model_spec_processor: str = "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        torch_dtype: str = "bfloat16",
    ):
        super().__init__()

        self.model: Mistral3ForConditionalGeneration = Mistral3ForConditionalGeneration.from_pretrained(
            model_spec,
            torch_dtype=getattr(torch, torch_dtype),
        )
        self.processor = AutoProcessor.from_pretrained(model_spec_processor, use_fast=False)
        self.yes_token, self.no_token = self.processor.tokenizer.encode(
            ["yes", "no"], add_special_tokens=False
        )

        self.max_length = MAX_LENGTH
        self.upsampling_max_image_size = UPSAMPLING_MAX_IMAGE_SIZE

        self.nsfw_classifier = pipeline("image-classification", model="Falconsai/nsfw_image_detection")

    def _validate_and_process_images(
        self, img: list[list[Image.Image]] | list[Image.Image]
    ) -> list[list[Image.Image]]:
        if not img:
            return []

        if isinstance(img[0], Image.Image):
            img = [[im] for im in img]

        img = [[concatenate_images(img_i)] if len(img_i) > 1 else img_i for img_i in img]
        img = [[cap_pixels(img_i, self.upsampling_max_image_size) for img_i in img_i] for img_i in img]
        return img

    def format_input(
        self,
        txt: list[str],
        system_message: str = SYSTEM_MESSAGE,
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
    ) -> list[list[dict]]:
        cleaned_txt = [prompt.replace("[IMG]", "") for prompt in txt]

        if img is None or len(img) == 0:
            return [
                [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": system_message,
                            }
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt,
                            }
                        ],
                    },
                ]
                for prompt in cleaned_txt
            ]

        processed_images = self._validate_and_process_images(img)

        return [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": system_message,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        *[
                            {
                                "type": "image",
                                "image": image,
                            }
                            for image in img_list
                        ],
                    ],
                },
            ]
            for prompt, img_list in zip(cleaned_txt, processed_images)
        ]

    def upsample_prompt_t2i(self, txt: list[str]) -> list[str]:
        chat = self.format_input(txt, system_message=SYSTEM_MESSAGE_UPSAMPLING_T2I)
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.15,
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        upsampled_prompts = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return upsampled_prompts

    def upsample_prompt_i2i(self, txt: list[str], img: list[Image.Image] | list[list[Image.Image]]) -> list[str]:
        chat = self.format_input(txt, system_message=SYSTEM_MESSAGE_UPSAMPLING_I2I, img=img)
        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.15,
        )

        generate_ids = generate_ids[:, inputs["input_ids"].shape[1] :]
        upsampled_prompts = self.processor.batch_decode(
            generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )

        return upsampled_prompts

    def upsample_prompt(
        self, txt: list[str], img: list[Image.Image] | list[list[Image.Image]] | None = None
    ) -> list[str]:
        if img is None:
            return self.upsample_prompt_t2i(txt)
        else:
            return self.upsample_prompt_i2i(txt, img)

    @torch.no_grad()
    def forward(
        self,
        txt: list[str],
        img: list[Image.Image] | list[list[Image.Image]] | None = None,
        system_message: str = SYSTEM_MESSAGE,
    ) -> torch.Tensor:
        chat = self.format_input(txt, system_message=system_message, img=img)

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )

        input_ids = inputs["input_ids"].to(self.model.device)
        attention_mask = inputs["attention_mask"].to(self.model.device)

        if img is not None:
            pixel_values = inputs["pixel_values"].to(device=self.model.device, dtype=self.model.dtype)
            image_sizes = inputs["image_sizes"].to(self.model.device)
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_sizes=image_sizes,
                output_hidden_states=True,
                use_cache=False,
            )
        else:
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_MISTRAL], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    def yes_no_logit_processor(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        scores_yes_token = scores[:, self.yes_token].clone()
        scores_no_token = scores[:, self.no_token].clone()
        scores_min = scores.min()
        scores[:, :] = scores_min - 1
        scores[:, self.yes_token] = scores_yes_token
        scores[:, self.no_token] = scores_no_token
        return scores

    def test_image(self, image: Image.Image | str | Path | torch.Tensor) -> bool:
        if isinstance(image, torch.Tensor):
            image = rearrange(image[0].clamp(-1.0, 1.0), "c h w -> h w c")
            image = Image.fromarray((127.5 * (image + 1.0)).cpu().byte().numpy())
        elif isinstance(image, (str, Path)):
            image = Image.open(image)

        classification = next(c for c in self.nsfw_classifier(image) if c["label"] == "nsfw")
        if classification["score"] > NSFW_THRESHOLD:
            return True

        w, h = image.size
        f = (512**2 / (w * h)) ** 0.5
        image = image.resize((int(f * w), int(f * h)))

        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY,
                    },
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": PROMPT_IMAGE_INTEGRITY_FOLLOW_UP,
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=self.model.dtype)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )

        return generate_ids[0, -1].item() == self.yes_token

    def test_txt(self, txt: str) -> bool:
        chat = [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT_CONTENT_FILTER,
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": PROMPT_TEXT_INTEGRITY.format(prompt=txt),
                    },
                ],
            },
        ]

        inputs = self.processor.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generate_ids = self.model.generate(
            **inputs,
            max_new_tokens=1,
            logits_processor=[self.yes_no_logit_processor],
            do_sample=False,
        )
        return generate_ids[0, -1].item() == self.yes_token


class Qwen3Embedder(nn.Module):
    def __init__(
        self,
        model_spec: str,
        tokenizer_spec: str | None = None,
        device: str | torch.device = "cuda",
    ):
        super().__init__()

        print(f"Loading Qwen3 model weights from: {model_spec}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_spec,
            torch_dtype=None,
            device_map=str(device),
        )

        if tokenizer_spec is None:
            if os.path.exists(model_spec):
                parent_dir = os.path.dirname(os.path.abspath(model_spec))
                sibling_tokenizer = os.path.join(parent_dir, "tokenizer")
                if os.path.exists(sibling_tokenizer):
                    tokenizer_spec = sibling_tokenizer
                else:
                    tokenizer_spec = model_spec
            else:
                tokenizer_spec = model_spec

        print(f"Loading Qwen3 tokenizer from: {tokenizer_spec}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_spec)
        self.max_length = MAX_LENGTH

    @torch.no_grad()
    def forward(self, txt: list[str] | str):
        if isinstance(txt, str):
            txt = [txt]

        all_input_ids = []
        all_attention_masks = []

        for prompt in txt:
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            model_inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            )

            all_input_ids.append(model_inputs["input_ids"])
            all_attention_masks.append(model_inputs["attention_mask"])

        input_ids = torch.cat(all_input_ids, dim=0).to(self.model.device)
        attention_mask = torch.cat(all_attention_masks, dim=0).to(self.model.device)

        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        out = torch.stack([output.hidden_states[k] for k in OUTPUT_LAYERS_QWEN3], dim=1)
        return rearrange(out, "b c l d -> b l (c d)")

    def test_txt(self, txt: str) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support text testing")

    def test_image(self, image) -> bool:
        raise NotImplementedError("Qwen3Embedder does not support image testing")

    def upsample_prompt(self, txt: list[str], img=None, **kwargs) -> list[str]:
        raise NotImplementedError("Qwen3Embedder does not support upsampling")


def load_mistral_small_embedder(device: str | torch.device = "cuda") -> Mistral3SmallEmbedder:
    return Mistral3SmallEmbedder().to(device)


def load_qwen3_embedder(variant: str, device: str | torch.device = "cuda"):
    env_var_key = f"QWEN3_{variant.upper()}_MODEL_PATH"
    model_spec = f"Qwen/Qwen3-{variant}-FP8"
    tokenizer_spec = None

    if env_var_key in os.environ and os.path.exists(os.environ[env_var_key]):
        model_spec = os.environ[env_var_key]
        print(f"Loading Qwen3 text encoder from env {env_var_key}: {model_spec}")
    elif "TEXT_ENCODER_PATH" in os.environ and os.path.exists(os.environ["TEXT_ENCODER_PATH"]):
        model_spec = os.environ["TEXT_ENCODER_PATH"]
        print(f"Loading Qwen3 text encoder from TEXT_ENCODER_PATH: {model_spec}")
    else:
        from .util import find_persistent_data_root
        p_root = find_persistent_data_root()
        if p_root:
            candidates = [
                os.path.join(p_root, "text_encoder"),
                p_root,
            ]
            for cp in candidates:
                if os.path.exists(cp) and (
                    os.path.exists(os.path.join(cp, "config.json"))
                    or os.path.exists(os.path.join(cp, "model.safetensors.index.json"))
                ):
                    model_spec = cp
                    print(f"Detected local Qwen3 weights at: {cp}")
                    break

    return Qwen3Embedder(model_spec=model_spec, tokenizer_spec=tokenizer_spec, device=device)
