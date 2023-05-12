
import dash
from dash import Dash, dcc, html
import base64
import cv2
import numpy as np
from dash.dependencies import Input, Output
from detection_methods import ImageProcessor, ContourDetector
import pandas as pd

app = dash.Dash()

app.layout = html.Div([
    dcc.Upload(
        id='upload-image',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        # Allow multiple files to be uploaded
        multiple=False
    ),
    html.Div(id='image-container'),


])


@app.callback(
    Output('image-container', 'children'),
    Input('upload-image', 'contents')
)
def analyze_image(contents):
    if contents is not None:
        img_processor = ImageProcessor()
        img_processor.set_image_data(contents)
        img_processor.set_original_image(img_processor.mri_img.copy())

        img_processor.threshold_image()
        img_processor.flood_fill()
        img_processor.auto_canny()

        brain_area = img_processor.find_brain_area()

        contour_detector = ContourDetector(img_processor.canny)

        contour_detector.find_contours()
        contour_detector.compute_cross_sectional_area()
        contour_detector.compute_tumor_severity()
        contour_detector.draw_contours(img_processor.mri_img)

        # -----------------------------------------------------

        contour_detector.extract_contour_tumors(img_processor.mri_img)
        contour_detector.write_extracted_contours_file()

        # -----------------------------------------------------


        # Encode the original image as base64
        _, buffer = cv2.imencode('.png', img_processor.original_img)
        original_data = base64.b64encode(buffer).decode('utf-8')

        # Encode the image with contours as base64
        _, buffer = cv2.imencode('.png', img_processor.mri_img)
        contour_data = base64.b64encode(buffer).decode('utf-8')

        # Display the image with contours
        if np.any(brain_area == 0):
            tumor_brain_ratio = float('0')
        else:
            tumor_brain_ratio = round(contour_detector.area / brain_area, 2)

        if np.isscalar(brain_area):
            percent = round((contour_detector.area / brain_area) * 100, 2)
        else:
            percent = float('nan')

        return [
            html.Div([
                html.H2('Original Image'),
                html.Img(src='data:image/png;base64,{}'.format(original_data))
            ], style={'float': 'left', 'width': '50%'}),

            html.Div([
                html.H2('Image with Contours'),
                html.Img(src='data:image/png;base64,{}'.format(contour_data))
            ], style={'float': 'right', 'width': '50%'}),


            html.H4('Results:'),
            html.P('Number of tumors detected: {}'.format(contour_detector.num_tumors)),
            html.P('The cross-sectional area of the tumor(s) mass is approximately: {}'.format(contour_detector.area)),
            html.P('Tumor:Brain area ratio: {}'.format(tumor_brain_ratio)),
            html.P(
                'The tumor occupies approximately {}% of the total cross-sectional area of the brain'.format(percent)),
            html.P('Severity of Tumor Growth: {}'.format(contour_detector.severity))
        ]

if __name__ == '__main__':
    app.run_server(debug=True)
