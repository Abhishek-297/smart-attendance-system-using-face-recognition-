from django.db import models
from django.utils import timezone
from django.contrib.auth.models import User

class Student(models.Model):
    YEAR_CHOICES = (
        (1, '1st Year'),
        (2, '2nd Year'),
        (3, '3rd Year'),
        (4, '4th Year'),
    )
    
    student_id = models.CharField(max_length=20, unique=True)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    email = models.EmailField(unique=True)
    year = models.IntegerField(choices=YEAR_CHOICES, default=1)
    face_image = models.ImageField(upload_to='faces/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def save(self, *args, **kwargs):
        # Auto-generate student_id in format: year + UG + counting_number (e.g., 1UG0001)
        if not self.student_id:
            # Get the last student ID for the same year
            last_student = Student.objects.filter(year=self.year).order_by('-student_id').first()
            
            if last_student and last_student.student_id.startswith(f"{self.year}UG"):
                try:
                    # Extract the numeric part and increment
                    last_num = int(last_student.student_id.split("UG")[1])
                    new_num = last_num + 1
                except (ValueError, IndexError):
                    # Fallback if parsing fails
                    new_num = 1
            else:
                # First student for this year
                new_num = 1
                
            # Format with leading zeros (4 digits)
            self.student_id = f"{self.year}UG{new_num:04d}"
        
        # Handle face image path
        if self.face_image:
            if self.year == 1:
                year_folder = "1st_year"
            elif self.year == 2:
                year_folder = "2nd_year"
            elif self.year == 3:
                year_folder = "3rd_year"
            else:
                year_folder = f"{self.year}th_year"
            
            timestamp = int(timezone.now().timestamp())
            new_path = f'faces/{year_folder}/{self.student_id}_{timestamp}.jpg'
            
            if not self.pk or not Student.objects.get(pk=self.pk).face_image:
                self.face_image.name = new_path
                
        super().save(*args, **kwargs)
    
    def __str__(self):
        year_str = f"{self.year}st" if self.year == 1 else f"{self.year}nd" if self.year == 2 else f"{self.year}rd" if self.year == 3 else f"{self.year}th"
        return f"{self.student_id} - {self.first_name} {self.last_name} ({year_str} Year)"

class Course(models.Model):
    YEAR_CHOICES = (
        (1, '1st Year'),
        (2, '2nd Year'),
        (3, '3rd Year'),
        (4, '4th Year'),
    )
    
    course_code = models.CharField(max_length=20, unique=True)
    course_name = models.CharField(max_length=100)
    year = models.IntegerField(choices=YEAR_CHOICES, default=1)
    
    def __str__(self):
        year_str = f"{self.year}st" if self.year == 1 else f"{self.year}nd" if self.year == 2 else f"{self.year}rd" if self.year == 3 else f"{self.year}th"
        return f"{self.course_code} - {self.course_name} ({year_str} Year)"

class UserCourse(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    
    class Meta:
        unique_together = ('user', 'course')
        
    def __str__(self):
        return f"{self.user.username} - {self.course}"

class Attendance(models.Model):
    student = models.ForeignKey(Student, on_delete=models.CASCADE)
    course = models.ForeignKey(Course, on_delete=models.CASCADE)
    date = models.DateField(default=timezone.now)
    time = models.TimeField(default=timezone.now)
    present = models.BooleanField(default=True)
    
    class Meta:
        unique_together = ('student', 'course', 'date')
    
    def __str__(self):
        return f"{self.student} - {self.course} - {self.date}"