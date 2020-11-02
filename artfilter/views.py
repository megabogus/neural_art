from django.http import JsonResponse, HttpResponse, Http404
from django.shortcuts import render, get_object_or_404, redirect
from .models import ArtFilter
from scipy.misc import imsave
from PIL import Image

# Create your views here.

def art_filter(request, filter_id):
    af = get_object_or_404(ArtFilter, pk=filter_id)
    best, best_loss = af.run_style_transfer(num_iterations=20)
    #imsave('%s_output.png' % af.pk, best)
    Image.fromarray(best)
    return JsonResponse({'status': 'ok',})

