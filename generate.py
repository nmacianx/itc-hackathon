from flask import Flask, jsonify
import boto3
import os
import numpy as np
from PIL import Image
from uuid import uuid4
from utils.utils_stylegan2 import convert_images_to_uint8
from stylegan2_generator import StyleGan2Generator


app = Flask(__name__)
TRUNCATION_PSI = 0.3
s3 = None
gen = None
w_avg = None


@app.route('/')
def generate():
    rnd = np.random.RandomState()
    z = rnd.randn(1, 512).astype('float32')
    dlatents = gen.mapping_network(z)
    dlatents = w_avg + (dlatents - w_avg) * TRUNCATION_PSI
    out = gen.synthesis_network(dlatents)

    img = convert_images_to_uint8(out, nchw_to_nhwc=True, uint8_cast=True)
    img = Image.fromarray(img.numpy()[0], 'RGB')
    filename = '{}.png'.format(uuid4())
    img.save(filename)

    with open(filename, 'rb') as f:
        s3.upload_fileobj(f, "itc-hackathon-group2", filename, ExtraArgs={
            'ACL': 'public-read'
        })

    url = 'https://itc-hackathon-group2.s3.eu-central-1.amazonaws.com/{}'.format(
        filename)
    return jsonify(url)


if __name__ == '__main__':
    port = os.environ.get('PORT')

    aws_access_key_id = os.environ['AWS_ACCESS_KEY_ID']
    aws_secret_access_key = os.environ['AWS_SECRET_ACCESS_KEY']
    s3 = boto3.client('s3',
                      aws_access_key_id=aws_access_key_id,
                      aws_secret_access_key=aws_secret_access_key)

    gen = StyleGan2Generator(weights='cat', impl='ref', gpu=False)
    w_avg = np.load('weights/cat_dlatent_avg.npy')

    if port:
        app.run(host='0.0.0.0', port=int(port))
    else:
        app.run()
