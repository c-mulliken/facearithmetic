import FreeSimpleGUI as sg
from PIL import Image
import numpy as np
import torch
import inference
import io

# model definitions
device = torch.device('cuda')
model = inference.StyleGan()

# helper functions
def refresh_images():
    latent_codes = [torch.randn([1, model.G.z_dim], device=device) for _ in range(25)]
    imgs = np.array(model.generate_imgs(latent_codes, truncation_psi=0.7))
    imgs = imgs.reshape((5, 5, 256, 256, 3)).transpose(0, 1, 4, 2, 3)
    latent_codes = torch.stack(latent_codes).reshape(5, 5, model.G.z_dim)
    return imgs.astype(np.uint8), latent_codes

def get_addition_panel():
    return [
        [sg.Text("Latent Code Addition", font=("Helvetica", 14))],
        [sg.Button("Image 1", key="ADD_1", size=(10, 1), disabled=False),
         sg.Text("+"),
         sg.Button("Image 2", key="ADD_2", size=(10, 1), disabled=False),
         sg.Text("="),
         sg.Button("Result", key="ADD_RESULT", size=(10, 1), disabled=True)],
        [sg.Button("Calculate Sum", key="CALCULATE_SUM"),
         sg.Button("Refresh Workbench", key="REFRESH_WORK")],
    ]


image_array, latent_codes = refresh_images()


# Function to convert NumPy array to PNG image data
def numpy_to_png(image_np):
    image = Image.fromarray(image_np.transpose(1, 2, 0))  # Convert from (3, H, W) to (H, W, 3)
    image = image.resize((128, 128))
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        return output.getvalue()

status_text = sg.Text("", size=(20, 1), key="STATUS", justification="center", font=("Helvetica", 14))
button_row = [sg.Button('Refresh Images', key='REFRESH')]
image_grid = [
    [
        sg.Button(
            image_data=numpy_to_png(image_array[row, col]),
            key=f"IMAGE_{row}_{col}",
            button_color=(sg.theme_background_color(), sg.theme_background_color()),
            border_width=0,
        )
        for col in range(5)
    ]
    for row in range(5)
]
right_panel_layout = [[sg.Text("Select an operation from above",
                               key="RIGHT_PANEL_PLACEHOLDER", font=("Helvetica", 12))],
                               [sg.Button("Add", key="ADD_MODE", size=(10, 1))]]
right_panel = sg.Column(right_panel_layout, key="RIGHT_PANEL",
                        vertical_alignment='top', element_justification='center')

layout = [
    [status_text],
    button_row,
    [sg.Column(image_grid), right_panel]
]

print(latent_codes[0, 0].shape)

# event handling variables
add_code_1 = None
add_code_2 = None
result_code = None
blank_1_clicked = False
blank_2_clicked = False
image_1_data = None
image_2_data = None
result_image_data = None

# Create the window
window = sg.Window("5x5 Image Grid from NumPy", layout, finalize=True)

# Event loop
while True:
    event, values = window.read()

    # Exit if the window is closed
    if event == sg.WINDOW_CLOSED:
        break
    
    if event == "ADD_1":
        blank_1_clicked = True
        blank_2_clicked = False

    if event == "ADD_2":
        blank_2_clicked = True
        blank_1_clicked = False

    if event == "CALCULATE_SUM":
        blank_1_clicked = False
        blank_2_clicked = False
        result_code = add_code_1 + add_code_2 
        # result_image = model.gen_img(result_code.reshape(1, 512), 0.7)
        # result_image = result_image.transpose(2, 0, 1).astype(np.uint8)
        # print(result_image.shape)
        # result_image = numpy_to_png(result_image)
        result_image = model.gen_pil(result_code.reshape(1, 512), 0.7)
        window["ADD_RESULT"].update(image_data=result_image, disabled=False)

    if event == "REFRESH_WORK":
        add_code_1 = None
        add_code_2 = None
        result_code = None
        blank_1_clicked = False
        blank_2_clicked = False
        image_1_data = None
        image_2_data = None
        result_image_data = None
        window["ADD_1"].update(image_data=None)
        window["ADD_2"].update(image_data=None)
        window["ADD_RESULT"].update(image_data=None, disabled=True)

    # Handle image button clicks
    if event.startswith("IMAGE_"):
        row, col = map(int, event.split("_")[1:])
        if blank_1_clicked:
            add_code_1 = latent_codes[row, col]
            print(image_array[row, col].dtype)
            print(image_array[row, col].shape)
            image_1_data = numpy_to_png(image_array[row, col])
            window["ADD_1"].update(image_data=image_1_data)
        elif blank_2_clicked:
            add_code_2 = latent_codes[row, col]
            image_2_data = numpy_to_png(image_array[row, col])
            window["ADD_2"].update(image_data=image_2_data)
        print(f"Image at row {row}, column {col} clicked!")
        print(f'{event}')
    
    if event == 'REFRESH':
        window["STATUS"].update("Refreshing...")
        window.refresh()  # Ensure UI updates immediately

        image_array, latent_codes = refresh_images()

        for row in range(5):
            for col in range(5):
                window[f'IMAGE_{row}_{col}'].update(image_data=numpy_to_png(image_array[row, col]))

        window['STATUS'].update('')

    if event == 'ADD_MODE':
        window["RIGHT_PANEL"].update(visible=False)
        right_panel_layout = get_addition_panel()
        window.extend_layout(window["RIGHT_PANEL"], right_panel_layout)
        window["RIGHT_PANEL"].update(visible=True)

# Close the window
window.close()
