import re
import base64
import numpy as np
from PIL import Image
from io import BytesIO
from keras.models import load_model

from django.http import HttpResponse, JsonResponse
from django.views.generic import TemplateView
from django.views.decorators.csrf import csrf_exempt

model = load_model('mnist.h5')


class PredictView(TemplateView):
    template_name = 'home.html'


@csrf_exempt
def predict(request):
    img = base64_img_to_array(request.POST['imgBase64'])

    result = model.predict([img])[0]
    prediction, accuracy = np.argmax(result), max(result)

    return JsonResponse({
        'prediction': int(prediction),
        'accuracy': float(accuracy),
    })


@csrf_exempt
def train(request):
    digit = base64_img_to_array(request.POST['imgBase64'])
    digit_answer = np.zeros(10)
    digit_answer[int(request.POST['digit'])] = 1
    x_train = [digit]
    y_train = np.array([digit_answer])

    model.fit(x_train, y_train, batch_size=128)

    return HttpResponse(status=204)


def base64_img_to_array(img):
    img_str = re.search(r'base64,(.*)', img).group(1)
    img_decoded = BytesIO(base64.b64decode(img_str))
    img = Image.open(img_decoded)

    img = img.resize((28, 28))
    img = img.convert('L')
    arr = np.array(img)

    arr = arr.reshape((1, 28, 28, 1))
    arr = arr / 255.0

    return arr
