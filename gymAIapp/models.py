from django.db import models
import json

# Create your models here.

class Contact(models.Model):
    name=models.CharField(max_length=25)
    email=models.EmailField()
    phonenumber=models.CharField(max_length=12)
    description=models.TextField()

    def __str__(self):
        return self.email

class BicepCurlResult(models.Model):
    analysis_time = models.DateTimeField(auto_now_add=True, help_text="Timestamp for when the analysis was conducted")
    total_curls = models.IntegerField(help_text="Total number of curls detected")
    form_errors = models.TextField(blank=True, default='[]', help_text="JSON-encoded list of detected form errors")
    feedback = models.TextField(blank=True, help_text="Summary or detailed feedback on the bicep curl analysis")
    
    # New field for the processed video filename
    processed_video_filename = models.CharField(max_length=255, blank=True, null=True, help_text="Filename of the processed video")

    def get_form_errors(self):
        """Return list of form errors."""
        return json.loads(self.form_errors)

    def set_form_errors(self, errors_list):
        """Save list of form errors as JSON."""
        self.form_errors = json.dumps(errors_list)

    class Meta:
        ordering = ['-analysis_time']

    def __str__(self):
        return f"Bicep Curl Analysis at {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
class SquatResult(models.Model):
    analysis_time = models.DateTimeField(auto_now_add=True, help_text="Timestamp for when the analysis was conducted")
    total_squats = models.IntegerField(help_text="Total number of squats detected")
    form_errors = models.TextField(blank=True, default='[]', help_text="JSON-encoded list of detected form errors")
    feedback = models.TextField(blank=True, help_text="Summary or detailed feedback on the squat analysis")
    
    # New field for the processed video filename
    processed_video_filename = models.CharField(max_length=255, blank=True, null=True, help_text="Filename of the processed video")

    def get_form_errors(self):
        """Return list of form errors."""
        return json.loads(self.form_errors)

    def set_form_errors(self, errors_list):
        """Save list of form errors as JSON."""
        self.form_errors = json.dumps(errors_list)

    class Meta:
        ordering = ['-analysis_time']

    def __str__(self):
        return f"Squat Analysis at {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
class ShoulderLateralRaiseResult(models.Model):
    analysis_time = models.DateTimeField(auto_now_add=True, help_text="Timestamp for when the analysis was conducted")
    total_reps = models.IntegerField(help_text="Total number of shoulder lateral raises detected")
    form_errors = models.TextField(blank=True, default='[]', help_text="JSON-encoded list of detected form errors")
    feedback = models.TextField(blank=True, help_text="Summary or detailed feedback on the shoulder lateral raise analysis")
    
    # Field for the processed video filename
    processed_video_filename = models.CharField(max_length=255, blank=True, null=True, help_text="Filename of the processed video")

    def get_form_errors(self):
        """Return list of form errors."""
        return json.loads(self.form_errors)

    def set_form_errors(self, errors_list):
        """Save list of form errors as JSON."""
        self.form_errors = json.dumps(errors_list)

    class Meta:
        ordering = ['-analysis_time']

    def __str__(self):
        return f"Shoulder Lateral Raise Analysis at {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
class BackPullDownResult(models.Model):
    analysis_time = models.DateTimeField(auto_now_add=True, help_text="Timestamp for when the analysis was conducted")
    total_reps = models.IntegerField(help_text="Total number of back pull-downs detected")
    form_errors = models.TextField(blank=True, default='[]', help_text="JSON-encoded list of detected form errors")
    feedback = models.TextField(blank=True, help_text="Summary or detailed feedback on the back pull-down analysis")
    
    # Field for the processed video filename
    processed_video_filename = models.CharField(max_length=255, blank=True, null=True, help_text="Filename of the processed video")

    def get_form_errors(self):
        """Return list of form errors."""
        return json.loads(self.form_errors)

    def set_form_errors(self, errors_list):
        """Save list of form errors as JSON."""
        self.form_errors = json.dumps(errors_list)

    class Meta:
        ordering = ['-analysis_time']

    def __str__(self):
        return f"Back Pull-Down Analysis at {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}"
    
class ChestCablePullResult(models.Model):
    analysis_time = models.DateTimeField(auto_now_add=True, help_text="Timestamp for when the analysis was conducted")
    total_reps = models.IntegerField(help_text="Total number of chest cable pulls detected")
    form_errors = models.TextField(blank=True, default='[]', help_text="JSON-encoded list of detected form errors")
    feedback = models.TextField(blank=True, help_text="Summary or detailed feedback on the chest cable pull analysis")
    
    # Field for the processed video filename
    processed_video_filename = models.CharField(max_length=255, blank=True, null=True, help_text="Filename of the processed video")

    def get_form_errors(self):
        """Return list of form errors."""
        return json.loads(self.form_errors)

    def set_form_errors(self, errors_list):
        """Save list of form errors as JSON."""
        self.form_errors = json.dumps(errors_list)

    class Meta:
        ordering = ['-analysis_time']

    def __str__(self):
        return f"Chest Cable Pull Analysis at {self.analysis_time.strftime('%Y-%m-%d %H:%M:%S')}"
