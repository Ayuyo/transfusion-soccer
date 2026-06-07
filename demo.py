"""
demo.py

This is the BYTE-LEVEL version (num_text_tokens=256, UTF-8 byte tokens).

Run:
    pip install gradio
    python demo.py \
        --checkpoint /content/drive/MyDrive/transfusion-soccer-checkpoints/checkpoint_step062000.pt \
        --share
"""

import argparse
import re
import torch
import torch.nn.functional as F
import gradio as gr
from pathlib import Path
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL
from transfusion_pytorch import Transfusion

from data.dataset import IMAGE_SIZE, decode_text


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .pt checkpoint")
    p.add_argument("--share",      action="store_true",
                   help="Create a public URL (lasts ~72 hours)")
    p.add_argument("--examples_dir", type=str, default=None,
                   help="Optional directory of example images to pre-populate")

    # Model config — must match training
    p.add_argument("--dim",      type=int, default=512)
    p.add_argument("--depth",    type=int, default=8)
    p.add_argument("--heads",    type=int, default=8)
    p.add_argument("--dim_head", type=int, default=64)

    return p.parse_args()


# Image transform (must match training)

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


# Text cleanup

def clean_commentary(text: str) -> str:
    """
    Strip artifacts from byte-level decoding:
    - Invalid UTF-8 replacement chars (U+FFFD)
    - The recurring "98298" generation artifact
    - Leading/trailing whitespace and punctuation
    """
    text = text.replace("\ufffd", "")
    text = re.sub(r"\s*98298\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip().lstrip(".,;:|/\\-").strip()
    return text


# Model loading (run once at startup)

class CommentaryGenerator:
    """Wraps the VAE + Transfusion model and runs inference per call."""

    def __init__(self, args, device):
        self.device = device

        print("Loading VAE...")
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
        self.vae = self.vae.to(device, dtype=torch.bfloat16)
        self.vae.requires_grad_(False)
        self.vae.eval()

        print("Loading Transfusion model...")
        self.model = Transfusion(
            num_text_tokens        = 256,    # byte-level
            dim_latent             = 4,
            channel_first_latent   = True,
            modality_default_shape = (32, 32),
            modality_num_dim       = 2,
            add_pos_emb            = True,
            transformer=dict(
                dim     = args.dim,
                depth   = args.depth,
                heads   = args.heads,
                dim_head= args.dim_head,
            )
        ).to(device)

        print(f"Loading checkpoint: {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location=device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        val_loss = ckpt.get("val_loss", None)
        self.checkpoint_info = (
            f"Step {ckpt.get('step', '?')}"
            + (f" • val_loss {val_loss:.4f}" if isinstance(val_loss, float) else "")
        )
        print(f"  {self.checkpoint_info}")

    @torch.no_grad()
    def generate(self, pil_image, temperature, cfg_scale,
                 max_tokens, n_samples):
        """Generate commentary samples from a PIL image."""
        if pil_image is None:
            return "Please upload an image first."

        # Preprocess
        img    = pil_image.convert("RGB")
        tensor = IMAGE_TRANSFORM(img).unsqueeze(0).to(self.device, dtype=torch.bfloat16)

        # Encode through VAE
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            latent = self.vae.encode(tensor).latent_dist.sample()
            latent = latent * self.vae.config.scaling_factor
            latent = F.avg_pool2d(latent, kernel_size=2)

        # Generate samples
        results = []
        for i in range(int(n_samples)):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = self.model.sample(
                    prompt=(0, latent[0]),
                    max_length=int(max_tokens),
                    text_temperature=float(temperature),
                    text_min_p=0.1,
                    cfg_scale=float(cfg_scale),
                )

            if isinstance(output, list):
                text_segments = []
                for item in output:
                    if isinstance(item, torch.Tensor) and item.dtype in (torch.long, torch.int):
                        text_segments.append(item.flatten())
                if not text_segments:
                    results.append(f"Sample {i+1}: <no text generated>")
                    continue
                text_tokens = torch.cat(text_segments)
            else:
                text_tokens = output

            commentary = decode_text(text_tokens)
            commentary = clean_commentary(commentary)
            results.append(f"Sample {i+1}:\n{commentary}")

        return "\n\n".join(results)


# Gradio interface

def build_interface(generator):
    css = """
    .gradio-container { max-width: 1100px !important; }
    .gr-button-primary { background: linear-gradient(90deg,#0b6e4f,#1d8a5e) !important;
                         border: none !important; }
    #title { text-align: center; }
    """

    with gr.Blocks(css=css, title="TransfusionSports — Soccer Commentary") as demo:
        gr.HTML("""
        <div id="title">
            <h1> TransfusionSports</h1>
            <h3>Image → Commentary using Transfusion (Meta AI 2024)</h3>
            <p><i>Stanford CS 131 Final Project • SoccerNet-Caption</i></p>
        </div>
        """)

        gr.Markdown(f"**Model checkpoint:** {generator.checkpoint_info}")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    type="pil",
                    label="Soccer broadcast frame",
                    height=400,
                )

                with gr.Accordion("Generation settings", open=True):
                    temperature = gr.Slider(
                        minimum=0.1, maximum=1.5, value=0.2, step=0.1,
                        label="Temperature",
                        info="Lower = focused / deterministic, Higher = diverse / creative"
                    )
                    cfg_scale = gr.Slider(
                        minimum=1.0, maximum=15.0, value=10.0, step=0.5,
                        label="Classifier-free guidance scale",
                        info="Higher values force stronger image conditioning"
                    )
                    max_tokens = gr.Slider(
                        minimum=50, maximum=500, value=200, step=10,
                        label="Max commentary length (bytes)"
                    )
                    n_samples = gr.Slider(
                        minimum=1, maximum=5, value=3, step=1,
                        label="Number of samples to generate"
                    )

                generate_btn = gr.Button("Generate Commentary", variant="primary")

            with gr.Column(scale=1):
                output_text = gr.Textbox(
                    label="Generated Commentary",
                    lines=18,
                    placeholder="Upload an image and click Generate...",
                )

        generate_btn.click(
            fn=generator.generate,
            inputs=[image_input, temperature, cfg_scale, max_tokens, n_samples],
            outputs=output_text,
        )

        gr.Markdown("""
        ---
        ### How it works
        1. Frame is resized to 512×512 and normalized to [-1, 1]
        2. Encoded through a frozen Stable Diffusion VAE → (4, 64, 64) latent
        3. Downsampled to (4, 32, 32) and passed into Transfusion as a `(modality, tensor)` prompt
        4. The model autoregressively generates byte-level commentary, conditioned on the image latent
        5. Classifier-free guidance (CFG) scales the difference between conditioned and unconditioned predictions, forcing tighter visual grounding

        Built with [Transfusion-PyTorch](https://github.com/lucidrains/transfusion-pytorch)
        on [SoccerNet-Caption](https://www.soccer-net.org/).
        """)

    return demo



def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    generator = CommentaryGenerator(args, device)
    demo = build_interface(generator)
    demo.launch(share=args.share, show_error=True)


if __name__ == "__main__":
    main()
