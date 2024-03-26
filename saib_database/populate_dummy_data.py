import os
import django

# Configure settings for project
# Need to run this before models can be imported
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'saib_database.settings')
django.setup()

from benchmarks.models import AIAccelerator, BenchmarkResult  # Import models

def add_dummy_data():
    # Clear existing data
    AIAccelerator.objects.all().delete()
    BenchmarkResult.objects.all().delete()

    # AI Accelerators Data
    accelerators_data = [
        ('NVIDIA A100', 'NVIDIA', 'Ampere'),
        ('AMD Instinct MI100', 'AMD', 'CDNA'),
        ('Intel Habana Gaudi2', 'Intel', 'Gaudi2')
    ]

    # Create AI Accelerator objects
    for name, manufacturer, architecture in accelerators_data:
        accelerator = AIAccelerator.objects.create(name=name, manufacturer=manufacturer, architecture=architecture)

        # Add some dummy benchmark results for each accelerator
        BenchmarkResult.objects.create(accelerator=accelerator, benchmark_name='FP32 Throughput', score=3120.5)
        BenchmarkResult.objects.create(accelerator=accelerator, benchmark_name='FP16 Throughput', score=6241.0)

    print('Dummy data added successfully.')

if __name__ == '__main__':
    add_dummy_data()
