import time
from django.shortcuts import render, redirect
from django.http import HttpResponse, JsonResponse
from django.contrib import messages
from django.views.decorators.csrf import csrf_exempt
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.decorators import login_required
from django.core.files.base import ContentFile
from django.conf import settings

from .models import Student, Course, Attendance, UserCourse
from .utils.face_recognition import recognize_face, check_image_quality, enhance_image_for_face_detection

import base64
import os
import json
import cv2
import numpy as np
from datetime import datetime
import logging

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import Table, TableStyle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@login_required(login_url='login')
@csrf_exempt
def index(request):
    return render(request, 'attendance/index.html')




# This patch addresses issues with image upload and folder creation in register_student view

# Update the register_student view in views.py:

@login_required(login_url='login')
@csrf_exempt
def register_student(request):
    if request.method == 'POST':
        try:
            # Get form data
            first_name = request.POST.get('first_name')
            last_name = request.POST.get('last_name')
            student_id = request.POST.get('student_id', '')  # Make optional
            email = request.POST.get('email')
            year = request.POST.get('year')
            face_image_data = request.POST.get('face_image')
            
            # Validate required fields
            if not all([first_name, last_name, email, year, face_image_data]):
                messages.error(request, 'All fields except Student ID are required')
                return render(request, 'attendance/register.html', {'years': Student.YEAR_CHOICES})
            
            # Check if a student ID is provided, and if so, validate it
            if student_id:
                if Student.objects.filter(student_id=student_id).exists():
                    messages.error(request, f'Student ID {student_id} already exists')
                    return render(request, 'attendance/register.html', {'years': Student.YEAR_CHOICES})
            
            # Check if email exists
            if Student.objects.filter(email=email).exists():
                messages.error(request, 'Email already registered')
                return render(request, 'attendance/register.html', {'years': Student.YEAR_CHOICES})
                
            # Process image data
            format, imgstr = face_image_data.split(';base64,') 
            image_data = base64.b64decode(imgstr)
            
            # Create student record first without saving to get an auto-generated ID if needed
            student = Student(
                first_name=first_name,
                last_name=last_name,
                email=email,
                year=year
            )
            
            # Set student_id if provided, otherwise it will be auto-generated in save()
            if student_id:
                student.student_id = student_id
            
            # Determine year folder
            if int(year) == 1:
                year_folder = "1st_year"
            elif int(year) == 2:
                year_folder = "2nd_year"
            elif int(year) == 3:
                year_folder = "3rd_year"
            else:
                year_folder = f"{year}th_year"
            
            # Create directory paths for image storage - BEFORE saving the file
            folder_path = os.path.join(settings.MEDIA_ROOT, 'faces', year_folder)
            os.makedirs(folder_path, exist_ok=True)
            
            # Generate a unique filename
            timestamp = int(time.time())
            if student_id:
                file_name = f"{student_id}_{timestamp}.jpg"
            else:
                # Temporary name that will be updated after student.save()
                file_name = f"temp_{timestamp}.jpg"
            
            # Save image to filesystem
            image_path = os.path.join(folder_path, file_name)
            with open(image_path, 'wb') as f:
                f.write(image_data)
            
            # Create a relative path for the database
            relative_path = os.path.join('faces', year_folder, file_name)
            student.face_image.name = relative_path
            
            # Now save the student
            student.save()
            
            # If we didn't have a student ID before, rename the file with the new ID
            if not student_id:
                # Remove old temp file
                os.remove(image_path)
                
                # Create new filename with correct student ID
                new_file_name = f"{student.student_id}_{timestamp}.jpg"
                new_image_path = os.path.join(folder_path, new_file_name)
                
                # Write the image data again
                with open(new_image_path, 'wb') as f:
                    f.write(image_data)
                
                # Update the database record
                new_relative_path = os.path.join('faces', year_folder, new_file_name)
                student.face_image.name = new_relative_path
                student.save(update_fields=['face_image'])
            
            messages.success(request, f'Student {first_name} {last_name} registered successfully with ID {student.student_id}!')
            return redirect('register_student')
            
        except Exception as e:
            logger.error(f"Error registering student: {str(e)}")
            messages.error(request, f'Registration error: {str(e)}')
    
    # Get all available years for the template
    return render(request, 'attendance/register.html', {'years': Student.YEAR_CHOICES})



@login_required(login_url='login')
@csrf_exempt
def take_attendance(request):
    if request.method == 'GET':
        # Filter courses based on user permissions
        if request.user.is_staff:
            # Admins see all courses
            courses = Course.objects.all()
        else:
            # Regular users only see assigned courses
            user_courses = UserCourse.objects.filter(user=request.user).values_list('course_id', flat=True)
            courses = Course.objects.filter(id__in=user_courses)
            
            # If user has no assigned courses
            if not courses.exists():
                messages.warning(request, 'You do not have any assigned courses. Please contact an administrator.')
        
        # Group courses by year
        courses_by_year = {}
        for year_val, year_label in Course.YEAR_CHOICES:
            year_courses = courses.filter(year=year_val)
            if year_courses.exists():
                courses_by_year[year_label] = year_courses
        
        return render(request, 'attendance/attendance.html', {
            'courses_by_year': courses_by_year,
            'years': Course.YEAR_CHOICES
        })
       
    elif request.method == 'POST':
        try:
            # Parse JSON data from request body
            try:
                data = json.loads(request.body)
                face_image_data = data.get('face_image')
                course_id = data.get('course_id')
            except json.JSONDecodeError:
                logger.error("Invalid JSON in request body")
                return JsonResponse({'success': False, 'message': 'Invalid JSON data'})

            if not face_image_data or not course_id:
                logger.warning("Missing required data for attendance")
                return JsonResponse({'success': False, 'message': 'Missing required data'})

            # Process image
            try:
                format, imgstr = face_image_data.split(';base64,')
                nparr = np.frombuffer(base64.b64decode(imgstr), np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is None:
                    logger.warning("Invalid image data in attendance")
                    return JsonResponse({'success': False, 'message': 'Invalid image data'})
                
                # Check image quality
                quality = check_image_quality(img)
                if quality['issues']:
                    logger.info(f"Image quality issues: {quality['issues']}")
                    
                    # Try to enhance image if there are quality issues
                    img = enhance_image_for_face_detection(img)
                    
                # Convert BGR to RGB - critical for face_recognition
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except Exception as e:
                logger.error(f"Error processing image: {str(e)}")
                return JsonResponse({'success': False, 'message': f'Error processing image: {str(e)}'})

            # Get course and student data
            try:
                course = Course.objects.get(pk=course_id)
                
                # Filter students by the course's year
                students = Student.objects.filter(year=course.year)
                
                if not students:
                    logger.warning(f"No students found for year {course.year}")
                    return JsonResponse({'success': False, 'message': f'No registered students found for {course.get_year_display()}'})
                
                # Check if any students have face images
                students_with_images = [s for s in students if s.face_image]
                
                if not students_with_images:
                    logger.warning(f"No students with face images for year {course.year}")
                    return JsonResponse({'success': False, 'message': f'No student face data available for {course.get_year_display()}'})
                    
            except Course.DoesNotExist:
                logger.error(f"Course with ID {course_id} not found")
                return JsonResponse({'success': False, 'message': 'Course not found'})
            except Exception as e:
                logger.error(f"Error retrieving database data: {str(e)}")
                return JsonResponse({'success': False, 'message': f'Database error: {str(e)}'})

            # Try multiple face recognition attempts with different parameters
            face_locations, face_names = recognize_face(img_rgb, students, tolerance=0.6)
            
            # If no faces found or all unknown, try with more permissive tolerance
            if not face_names or all(name == "Unknown" for name in face_names):
                logger.info("First recognition attempt failed, trying with higher tolerance")
                face_locations, face_names = recognize_face(img_rgb, students, tolerance=0.65)
                
                # If still no luck, try one more time with enhanced image
                if not face_names or all(name == "Unknown" for name in face_names):
                    logger.info("Second recognition attempt failed, trying with enhanced image")
                    
                    # Apply additional enhancement
                    enhanced_img = enhance_image_for_face_detection(img)
                    enhanced_rgb = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2RGB)
                    
                    face_locations, face_names = recognize_face(enhanced_rgb, students, tolerance=0.7)

            if not face_names or all(name == "Unknown" for name in face_names):
                logger.warning("No registered students recognized after multiple attempts")
                return JsonResponse({'success': False, 'message': 'No registered student found in the image. Please ensure students are clearly visible and try again.'})

            # Mark attendance for recognized students
            recognized_students = []
            today = datetime.now().date()
            current_time = datetime.now().time()
            
            for student_id in face_names:
                if student_id != "Unknown":
                    try:
                        student = Student.objects.get(student_id=student_id)
                        
                        # Check if student is in the right year for this course
                        if student.year != course.year:
                            logger.warning(f"Student {student_id} is year {student.year} but course is for year {course.year}")
                            continue
                            
                        attendance, created = Attendance.objects.get_or_create(
                            student=student,
                            course=course,
                            date=today,
                            defaults={'time': current_time}
                        )

                        recognized_students.append({
                            'student_id': student.student_id,
                            'name': f"{student.first_name} {student.last_name}",
                            'year': student.get_year_display(),
                            'marked_now': created
                        })
                        
                        logger.info(f"Marked attendance for student: {student_id} in course: {course.course_code}")
                    except Student.DoesNotExist:
                        logger.error(f"Student with ID {student_id} not found")
                    except Exception as e:
                        logger.error(f"Error marking attendance: {str(e)}")

            return JsonResponse({'success': True, 'recognized_students': recognized_students})
        
        except Exception as e:
            logger.error(f"Unhandled error in take_attendance: {str(e)}")
            return JsonResponse({'success': False, 'message': f'Failed to take attendance: {str(e)}'}) 
                
        
    
@login_required(login_url='login')
@csrf_exempt
def attendance_reports(request):
    # Filter courses based on user permissions - moved outside method conditionals
    if request.user.is_staff:
        # Admins see all courses
        courses = Course.objects.all()
    else:
        # Regular users only see assigned courses
        user_courses = UserCourse.objects.filter(user=request.user).values_list('course_id', flat=True)
        courses = Course.objects.filter(id__in=user_courses)
        
        # If user has no assigned courses
        if not courses.exists():
            messages.warning(request, 'You do not have any assigned courses. Please contact an administrator.')
    
    # Group courses by year
    courses_by_year = {}
    for year_val, year_label in Course.YEAR_CHOICES:
        year_courses = courses.filter(year=year_val)
        if year_courses.exists():
            courses_by_year[year_label] = year_courses
    
    if request.method == 'GET':
        return render(request, 'attendance/reports.html', {
            'courses_by_year': courses_by_year,
            'years': Course.YEAR_CHOICES
        })
       
    if request.method == 'POST':
        try:
            course_id = request.POST.get('course_id')
            start_date = request.POST.get('start_date')
            end_date = request.POST.get('end_date')

           
            if not all([course_id, start_date, end_date]):
                messages.error(request, 'Missing required fields')
                return render(request, 'attendance/reports.html', {
                    'courses_by_year': courses_by_year,
                    'years': Course.YEAR_CHOICES
                })

            # Filter attendance records by course and date
            attendance_records = Attendance.objects.filter(
                course_id=course_id,
                date__range=[start_date, end_date]
            ).order_by('date')
            
            selected_course = Course.objects.get(pk=course_id)

            return render(request, 'attendance/reports.html', {
                'courses_by_year': courses_by_year,
                'years': Course.YEAR_CHOICES,
                'attendance_records': attendance_records,
                'selected_course': selected_course,
                'start_date': start_date,
                'end_date': end_date
            })
        except Course.DoesNotExist:
            messages.error(request, 'Course not found')
        except Exception as e:
            logger.error(f"Error generating report: {str(e)}")
            messages.error(request, f'Error generating report: {str(e)}')

    return render(request, 'attendance/reports.html', {
        'courses_by_year': courses_by_year,
        'years': Course.YEAR_CHOICES
    })


@csrf_exempt
def download_report_pdf(request):
    if request.method == 'GET':
        try:
            course_id = request.GET.get('course_id')
            start_date = request.GET.get('start_date')
            end_date = request.GET.get('end_date')

            if not all([course_id, start_date, end_date]):
                return HttpResponse('Missing required parameters', status=400)

            attendance_records = Attendance.objects.filter(
                course_id=course_id,
                date__range=[start_date, end_date]
            ).order_by('date')

            course = Course.objects.get(pk=course_id)

            response = HttpResponse(content_type='application/pdf')
            response['Content-Disposition'] = f'attachment; filename="attendance_report_{course.course_code}.pdf"'

            p = canvas.Canvas(response, pagesize=letter)
            width, height = letter

            # Page header
            p.setFont("Helvetica-Bold", 16)
            p.drawString(50, height - 50, f"Attendance Report: {course.course_name} ({course.course_code})")

            p.setFont("Helvetica", 12)
            p.drawString(50, height - 80, f"Period: {start_date} to {end_date}")
            p.drawString(50, height - 100, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

            # Table data
            data = [['Date', 'Student ID', 'Student Name', 'Status']]

            for record in attendance_records:
                data.append([
                    record.date.strftime('%Y-%m-%d'),
                    record.student.student_id,
                    f"{record.student.first_name} {record.student.last_name}",
                    'Present' if record.present else 'Absent'
                ])

            # Summary section
            total_records = len(attendance_records)
            present_count = sum(1 for record in attendance_records if record.present)
            
            # Create and style table
            table = Table(data, colWidths=[100, 100, 200, 80])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))

            table.wrapOn(p, width - 100, height)
            
            # Adjust table position based on number of records
            table_height = 20 * (len(data) + 1)
            table.drawOn(p, 50, height - 120 - table_height)
            
            # Add summary after table
            summary_y = height - 140 - table_height
            p.setFont("Helvetica-Bold", 12)
            p.drawString(50, summary_y, f"Summary:")
            p.setFont("Helvetica", 12)
            p.drawString(50, summary_y - 20, f"Total Records: {total_records}")
            p.drawString(50, summary_y - 40, f"Present: {present_count} ({int(present_count/total_records*100) if total_records else 0}%)")
            p.drawString(50, summary_y - 60, f"Absent: {total_records - present_count} ({int((total_records-present_count)/total_records*100) if total_records else 0}%)")

            p.showPage()
            p.save()
            return response
            
        except Exception as e:
            logger.error(f"Error downloading report: {str(e)}")
            return HttpResponse(f'Error generating PDF: {str(e)}', status=500)

@login_required
@csrf_exempt
def check_student_id_exists(request):
    """
    Checks if a student ID already exists in the database
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            student_id = data.get('student_id')
            
            if not student_id:
                return JsonResponse({'error': 'Student ID is required'}, status=400)
                
            exists = Student.objects.filter(student_id=student_id).exists()
            return JsonResponse({'exists': exists})
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            logger.error(f"Error checking student ID: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

@login_required
@csrf_exempt
def check_email_exists(request):
    """
    Checks if an email already exists in the database
    """
    if request.method == 'POST':
        try:
            data = json.loads(request.body)
            email = data.get('email')
            
            if not email:
                return JsonResponse({'error': 'Email is required'}, status=400)
                
            exists = Student.objects.filter(email=email).exists()
            return JsonResponse({'exists': exists})
            
        except json.JSONDecodeError:
            return JsonResponse({'error': 'Invalid JSON'}, status=400)
        except Exception as e:
            logger.error(f"Error checking email: {str(e)}")
            return JsonResponse({'error': str(e)}, status=500)
    
    return JsonResponse({'error': 'Only POST method is allowed'}, status=405)

@csrf_exempt
def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        login_type = request.POST.get('login_type', 'user')  # Default to user login

        if not all([username, password]):
            messages.error(request, 'Please provide both username and password')
            return render(request, 'attendance/login.html')

        user = authenticate(request, username=username, password=password)

        if user is not None:
            # For admin login, check if user is staff or superuser
            if login_type == 'admin' and not (user.is_staff or user.is_superuser):
                messages.error(request, 'You do not have admin privileges')
                return render(request, 'attendance/login.html')
            
            # For regular user login, you might want to check some other condition
            # or just allow any authenticated user
            
            # Remember me functionality
            if request.POST.get('remember_me'):
                # Session will last for 2 weeks
                request.session.set_expiry(1209600)
            else:
                # Session will expire when the user closes the browser
                request.session.set_expiry(0)
                
            login(request, user)
            
            # Redirect to appropriate dashboard based on login type
            if login_type == 'admin' and (user.is_staff or user.is_superuser):
                next_page = request.GET.get('next', 'index')
            else:
                next_page = request.GET.get('next', 'index')
                
            return redirect(next_page)
        else:
            login_type_display = 'Admin' if login_type == 'admin' else 'User'
            messages.error(request, f'Invalid {login_type_display} credentials')

    return render(request, 'attendance/login.html')


def user_logout(request):
    logout(request)
    messages.success(request, 'You have been logged out successfully')
    return redirect('login')
