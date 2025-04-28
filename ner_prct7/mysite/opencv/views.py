from django.shortcuts import get_object_or_404, render, redirect
from django.http import StreamingHttpResponse
import cv2
from .forms import *
from .models import *

def filter(request, id):
    video = get_object_or_404(Video, id=id)
    return render(request, template_name='opencv/filter.html', context={'video': video})


def video_filter(request, id):
    filter_type = request.GET.get('filter')
    return StreamingHttpResponse(stream_rgb_video(id, filter_type), content_type='multipart/x-mixed-replace; boundary=frame')


def stream_rgb_video(id, filter_type='RGB'):
    video = get_object_or_404(Video, id=id)
    video_path = video.video_file.path
    
    # Бесконечный перезапуск видео
    while True:
        # Читаем путь в cap
        cap = cv2.VideoCapture(video_path)

        while cap.isOpened():
            ret, img = cap.read()

            # Когда кадры закончились, выходим из цикла стрима кадров
            if not ret:
                break
            
            if filter_type == 'GRAY':
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            elif filter_type == 'HSV':
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            else:
                hsv = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_bytes = cv2.imencode('.jpg', hsv)[1].tobytes()
            cv2.waitKey(24)

            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')

        # Освобождаем ресурсы для cap
        cap.release()

"""
--frame  -  маркер начала нового кадра в потоке. Клиент, получающий этот поток
будет искать эту последовательность байтов, чтобы понять, где начинается новый кадр

Content-Type: ...  -  заголовок, указывающий на то, что данные будут изображением JPEG

image_bytes  -  массив байтов изображения (тело запроса)

/r/n  -  символы для разделения составляющих команды
"""
        
        


def video_list(request):
    videos = Video.objects.all()
    return render(request, template_name='opencv/video_list.html', context={'videos': videos})


def upload_video(request):
    if request.method == 'POST':
        form = VideoForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            return redirect('video_list')
    else:
        form = VideoForm()
    return render(request, template_name='opencv/upload_video.html', context={'form': form})


# def image_filter(request):
#     return StreamingHttpResponse(stream_image(), content_type='multipart/x-mixed-replace; boundary=frame')


# def stream_image():
#     img = cv2.imread('media/image.png')

#     # Преобразуем цвета из BGR в HSV
#     rgb = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Кодируем изображение в формат JPEG
#     image_bytes = cv2.imencode('.jpg', rgb)[1].tobytes()

#     # Формируем часть потока данных между кадрами (для отправки клиенту)
#     yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')