from predict import SchnellPredictor, DevPredictor


def schnell(prompt, task, control_image, num_outputs=1, seed=42):
    predictor = SchnellPredictor()
    predictor.setup()
    predictor.predict(
        prompt=prompt,
        task=task,
        control_image=control_image,
        num_outputs=num_outputs,
        num_inference_steps=8,
        guidance_scale=3.5,
        seed=seed,
        output_format="webp",
        output_quality=100,
    )


def dev(prompt, task, control_image, num_outputs=1, seed=42):
    predictor = DevPredictor()
    predictor.setup()
    predictor.predict(
        prompt=prompt,
        task=task,
        control_image=control_image,
        num_outputs=num_outputs,
        num_inference_steps=28,
        guidance_scale=3.5,
        seed=seed,
        output_format="webp",
        output_quality=100,
    )


def subject_512():
    prompt = "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat, holding a sign that reads 'Omini Control!'"
    control_image = "assets/penguin.jpg"
    task = "subject_512"
    schnell(prompt, task, control_image)


def subject_1024():
    prompt = "On the beach, a lady sits under a beach umbrella. She's wearing this shirt and has a big smile on her face, with her surfboard hehind her. The sun is setting in the background. The sky is a beautiful shade of orange and purple."
    control_image = "assets/tshirt.jpg"
    task = "subject_1024"
    schnell(prompt, task, control_image, seed=0, num_outputs=4)


def fill():
    prompt = (
        "A yellow book with the word 'OMINI' in large font on the cover. The text 'for FLUX' appears at the bottom."
    )
    control_image = "assets/book_masked.jpg"
    dev(prompt, "fill", control_image)


def spatial(task):
    prompt = "In a bright room. A cup of a coffee with some beans on the side. They are placed on a dark wooden table."
    control_image = "assets/coffee.png"
    dev(prompt, task, control_image)


def canny():
    spatial("canny")


def depth():
    spatial("depth")


def coloring():
    spatial("coloring")


def deblurring():
    spatial("deblurring")


if __name__ == "__main__":
    subject_1024()
