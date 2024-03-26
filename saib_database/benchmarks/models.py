from django.db import models

class AIAccelerator(models.Model):
    name = models.CharField(max_length=100)
    manufacturer = models.CharField(max_length=100)
    architecture = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class BenchmarkResult(models.Model):
    accelerator = models.ForeignKey(AIAccelerator, on_delete=models.CASCADE)
    benchmark_name = models.CharField(max_length=100)
    score = models.FloatField()

    def __str__(self):
        return f"{self.accelerator.name} - {self.benchmark_name}"
