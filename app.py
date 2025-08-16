
import gradio as gr
from video_process import add_mask
def ui():
    # with gr.Blocks(theme='NoCrypt/miku') as demo:
    with gr.Blocks() as demo:
        gr.Markdown(
            "## Hide Face Using Squid Game Masks",
            elem_classes="text-center"
        )
        mask_names=["Front Man Mask", "Guards Mask", "Red Mask", "Blue Mask"]
        dummy_examples=[["./assets/zuck.mp4"]]
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="Upload Video")
                mask_selector = gr.Dropdown(choices=mask_names, label="Select Mask")
                submit_btn = gr.Button("Apply Mask")

                with gr.Accordion('Mask Settings', open=False):
                    mask_up = gr.Slider(minimum=0, maximum=100, label="Mask Up (Forehead Side)", value=10)
                    mask_down = gr.Slider(minimum=0, maximum=100, label="Mask Down (Chin Side)", value=0)

            with gr.Column():
                output_video = gr.Video(label="Output Video")
                download_video = gr.File(label="Download Video")  

        inputs = [video_input, mask_selector, mask_up, mask_down]
        outputs = [output_video, download_video]

        submit_btn.click(add_mask, inputs=inputs, outputs=outputs)
        gr.Examples(examples=dummy_examples, 
                    inputs=[video_input] )#,
                    # cache_examples=True)
    return demo
import click
@click.command()
@click.option("--debug", is_flag=True, default=False, help="Enable debug mode.")
@click.option("--share", is_flag=True, default=False, help="Enable sharing of the interface.")
def main(debug, share):
# def main(debug=True, share=False):
    demo=ui()
    demo.launch()
    demo.queue().launch(debug=debug, share=share)
if __name__ == "__main__":
    main()    
