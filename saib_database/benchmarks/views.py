from django.shortcuts import render
from .models import AIAccelerator, BenchmarkResult

def index(request):
    return render(request, 'benchmarks/index.html')

def accelerators(request):
    # Fetch all AI accelerators and their benchmarks
    ai_accelerators = AIAccelerator.objects.all()
    benchmark_results = BenchmarkResult.objects.select_related('accelerator').all()
    context = {'ai_accelerators': ai_accelerators, 'benchmark_results': benchmark_results}
    return render(request, 'benchmarks/accelerators.html', context)
