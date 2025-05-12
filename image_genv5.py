import torch
import time
import numpy as np
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import gradio as gr

# Device config
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
def load_models():
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_scribble",
        torch_dtype=torch.float16
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to(device)

    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    return pipe

pipe = load_models()

STYLE_PRESETS = {
    "French Classical": "french classical interior, ornate paneling, gilded moldings, high ceilings, symmetry, soft beige and blue tones",
    "Warm Minimalism": "minimalist interior design, warm oak tones, soft textiles, indirect lighting, Scandinavian influence, cozy yet spacious",
    "Quiet Luxury": "luxury interior, neutral palette, bespoke furniture, subtle textures, natural lighting, timeless elegance",
    "Modern Loft": "loft interior, industrial elements, exposed brick, steel beams, modern furniture, large windows",
    "Coastal Contemporary": "coastal interior, light-filled spaces, natural materials, breezy fabrics, relaxed elegance"
}

def generate(drawing, prompt, style, steps=25, guidance=7.5, creativity=0.7):
    try:
        full_prompt = f"{STYLE_PRESETS[style]}, {prompt}, interior render, realistic materials, high detail, 8k"

        if isinstance(drawing, dict):
            drawing = drawing.get('image', drawing.get('background', None))
        if drawing is None:
            return None

        img = Image.fromarray(drawing.astype('uint8')) if isinstance(drawing, np.ndarray) else drawing
        img = img.convert("RGB").resize((512, 512))

        result = pipe(
            prompt=full_prompt,
            negative_prompt="blurry, low quality, distorted, bad proportions, cartoon, unrealistic",
            image=img,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            controlnet_conditioning_scale=float(creativity),
            generator=torch.Generator(device=device).manual_seed(int(time.time()))
        )

        return result.images[0]
    except Exception as e:
        print(f"Error: {e}")
        return None

def clear_sketch():
    return None

with gr.Blocks(title="Real-Time Image generator", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        """
        <div style='text-align: center; font-size: 34px; font-weight: bold;'>
        🎨 Real-Time Image generator<br>
        <span style='font-size: 22px;'>Draw your idea, enter a prompt, and watch the AI bring it to life!</span>
        </div>
        """,
        elem_id="title",
    )

    with gr.Row():
        with gr.Column():
            sketch = gr.ImageEditor(
                label="🖌 Draw or Upload an Image",
                type="pil",
                interactive=True,
                height=512,
                width=512,
                brush=gr.Brush(colors=["black", "blue", "red", "green"])
            )
            clear_btn = gr.Button("🩹 Clear Sketch")
            prompt = gr.Textbox(label="Prompt", placeholder="Describe what you want to create...", lines=2)
            style_dropdown = gr.Dropdown(label="Style Preset", choices=list(STYLE_PRESETS.keys()), value="Warm Minimalism")
            with gr.Accordion("⚙ Advanced Settings", open=False):
                steps = gr.Slider(5, 30, value=25, step=1, label="Steps")
                guidance = gr.Slider(1.0, 10.0, value=7.5, step=0.1, label="Guidance")
                creativity = gr.Slider(0.1, 1.0, value=0.7, step=0.1, label="Drawing Influence")
            generate_btn = gr.Button("🚀 Generate Image")
        with gr.Column():
          output = gr.Image(label="AI Output", interactive=False, width=512, height=512)
          download_btn = gr.Button("📅 Download Image")
          with gr.Row():
              status = gr.Textbox(label="Status", value="Waiting for input...", interactive=False)


    inputs = [sketch, prompt, style_dropdown, steps, guidance, creativity]
    generate_btn.click(lambda *_: "✨ Generating...", outputs=status, show_progress=False)
    generate_btn.click(fn=generate, inputs=inputs, outputs=output, show_progress=True)
    generate_btn.click(lambda: "✅ Done!", outputs=status, show_progress=False)
    clear_btn.click(fn=clear_sketch, outputs=sketch)
    download_btn.click(lambda img: img, inputs=output, outputs=output)

app.launch(share=True)