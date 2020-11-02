from django.http import JsonResponse, HttpResponse, Http404
from django.shortcuts import render, get_object_or_404, redirect
from .models import ArtFilter
# from scipy.misc import imsave
from PIL import Image

# Create your views here.

def art_filter(request, filter_id):
    af = get_object_or_404(ArtFilter, pk=filter_id)
    best, best_loss = af.run_train(num_iterations=1000)
    # imsave('%s_output.png' % af.pk, best)
    im = Image.fromarray(best)
    im.save('%s_output.jpg' % af.pk, 'JPEG')
    return JsonResponse({'status': 'ok',})

