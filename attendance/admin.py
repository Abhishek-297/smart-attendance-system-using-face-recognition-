# admin.py for your attendance app

from django.contrib import admin

from django.contrib.auth.models import User, Group
from django.contrib.auth.admin import UserAdmin, GroupAdmin

from django.shortcuts import render
from django.utils.html import format_html
from django.urls import path
from django.template.response import TemplateResponse
from django.db.models import Count
from django.http import HttpResponse
import csv
from datetime import datetime

from .models import Student, Course, Attendance, UserCourse

# ------------------ Custom Admin Site ------------------

class CustomAdminSite(admin.AdminSite):
    site_header = 'Smart Attendance System Administration'
    site_title = 'Smart Attendance Admin'
    index_title = 'Attendance Management Dashboard'

    def index(self, request, extra_context=None):
        app_list = self.get_app_list(request)
        student_count = Student.objects.count()
        course_count = Course.objects.count()
        attendance_count = Attendance.objects.count()
        present_count = Attendance.objects.filter(present=True).count()

        present_percentage = (present_count / attendance_count * 100) if attendance_count > 0 else 0
        recent_attendance = Attendance.objects.order_by('-date', '-time')[:10]
        
        # Get year-wise student counts
        year_counts = []
        for year_num, year_name in Student.YEAR_CHOICES:
            count = Student.objects.filter(year=year_num).count()
            year_counts.append({'year': year_name, 'count': count})

        context = {
            'app_list': app_list,
            'title': self.index_title,
            'student_count': student_count,
            'course_count': course_count,
            'attendance_count': attendance_count,
            'present_percentage': present_percentage,
            'recent_attendance': recent_attendance,
            'year_counts': year_counts,
            **(extra_context or {}),
        }

        request.current_app = self.name
        return TemplateResponse(request, 'admin/index.html', context)

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path('course_attendance_summary/', self.admin_view(self.course_attendance_summary),
                 name='course_attendance_summary'),
        ]
        return custom_urls + urls

    def course_attendance_summary(self, request):
        courses = Course.objects.all()
        course_data = []

        for course in courses:
            total_students = Student.objects.filter(year=course.year).count()
            attendance_dates = Attendance.objects.filter(course=course).values('date').distinct()
            present_counts = {}

            for date_obj in attendance_dates:
                date = date_obj['date']
                present_count = Attendance.objects.filter(course=course, date=date, present=True).count()
                present_counts[date.strftime('%Y-%m-%d')] = {
                    'count': present_count,
                    'percentage': (present_count / total_students) * 100 if total_students > 0 else 0
                }

            course_data.append({
                'course': course,
                'attendance_data': present_counts
            })

        context = {
            'title': 'Course Attendance Summary',
            'course_data': course_data,
        }
        return TemplateResponse(request, 'admin/course_attendance_summary.html', context)


admin_site = CustomAdminSite(name='admin')

# ------------------ Student Admin ------------------

class AttendanceInline(admin.TabularInline):
    model = Attendance
    extra = 0
    fields = ('course', 'date', 'time', 'present')
    readonly_fields = ('time',)

class StudentAdmin(admin.ModelAdmin):
    list_display = ('student_id', 'full_name', 'email', 'year_display', 'show_face_image', 'attendance_count')
    search_fields = ('student_id', 'first_name', 'last_name', 'email')
    list_filter = ('year', 'attendance__course', 'attendance__date')

    fieldsets = (
        ('Student Information', {
            'fields': ('student_id', 'first_name', 'last_name', 'email', 'year')
        }),
        ('Face Recognition Data', {
            'fields': ('face_image',),
            'classes': ('collapse',),
        }),
    )

    exclude = ('face_encoding',)
    inlines = [AttendanceInline]

    def full_name(self, obj):
        return f"{obj.first_name} {obj.last_name}"
    full_name.short_description = 'Full Name'
    
    def year_display(self, obj):
        return obj.get_year_display()
    year_display.short_description = 'Year'

    def show_face_image(self, obj):
      if obj.face_image and hasattr(obj.face_image, 'url'):
        try:
            return format_html('<img src="{}" width="50" height="50" />', obj.face_image.url)
        except Exception as e:
            return f"Error loading image: {str(e)}"
      return "No image"

    def attendance_count(self, obj):
        return obj.attendance_set.filter(present=True).count()
    attendance_count.short_description = 'Attendance Count'

    actions = ['export_as_csv']

    def export_as_csv(self, request, queryset):
        meta = self.model._meta
        field_names = [field.name for field in meta.fields if field.name != 'face_encoding']

        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename={meta}-{datetime.now().strftime("%Y%m%d")}.csv'
        writer = csv.writer(response)

        writer.writerow(field_names)
        for obj in queryset:
            row = [getattr(obj, field) for field in field_names]
            writer.writerow(row)

        return response
    export_as_csv.short_description = "Export selected students as CSV"

# ------------------ Course Admin ------------------

class CourseAdmin(admin.ModelAdmin):
    list_display = ('course_code', 'course_name', 'year_display', 'attendance_count', 'student_count')
    search_fields = ('course_code', 'course_name')
    list_filter = ('year',)
    
    def year_display(self, obj):
        return obj.get_year_display()
    year_display.short_description = 'Year'

    def attendance_count(self, obj):
        return Attendance.objects.filter(course=obj).count()
    attendance_count.short_description = 'Total Attendance Records'

    def student_count(self, obj):
        # Count students in the same year as the course
        return Student.objects.filter(year=obj.year).count()
    student_count.short_description = 'Eligible Students'

# ------------------ UserCourse Admin ------------------

class UserCourseAdmin(admin.ModelAdmin):
    list_display = ('user', 'course')
    list_filter = ('user', 'course', 'course__year')
    search_fields = ('user__username', 'course__course_name', 'course__course_code')

# ------------------ Attendance Admin ------------------

class AttendanceFilter(admin.SimpleListFilter):
    title = 'Attendance Status'
    parameter_name = 'attendance_status'

    def lookups(self, request, model_admin):
        return (
            ('present', 'Present'),
            ('absent', 'Absent'),
        )

    def queryset(self, request, queryset):
        if self.value() == 'present':
            return queryset.filter(present=True)
        if self.value() == 'absent':
            return queryset.filter(present=False)

class AttendanceAdmin(admin.ModelAdmin):
    list_display = ('student', 'course', 'date', 'time', 'present')
    list_filter = ('course', 'course__year', 'student__year', 'date', AttendanceFilter)
    search_fields = ('student__first_name', 'student__last_name', 'student__student_id', 'course__course_name')
    date_hierarchy = 'date'
    list_editable = ('present',)

    fieldsets = (
        (None, {
            'fields': ('student', 'course')
        }),
        ('Attendance Information', {
            'fields': ('date', 'time', 'present')
        }),
    )

    actions = ['mark_present', 'mark_absent', 'export_attendance_report']

    def mark_present(self, request, queryset):
        updated = queryset.update(present=True)
        self.message_user(request, f"{updated} attendance records marked as present.")
    mark_present.short_description = "Mark selected records as present"

    def mark_absent(self, request, queryset):
        updated = queryset.update(present=False)
        self.message_user(request, f"{updated} attendance records marked as absent.")
    mark_absent.short_description = "Mark selected records as absent"

    def export_attendance_report(self, request, queryset):
        response = HttpResponse(content_type='text/csv')
        response['Content-Disposition'] = f'attachment; filename=attendance_report_{datetime.now().strftime("%Y%m%d")}.csv'

        writer = csv.writer(response)
        writer.writerow(['Student ID', 'Student Name', 'Year', 'Course', 'Date', 'Time', 'Status'])

        for record in queryset:
            writer.writerow([
                record.student.student_id,
                f"{record.student.first_name} {record.student.last_name}",
                record.student.get_year_display(),
                record.course.course_name,
                record.date,
                record.time,
                'Present' if record.present else 'Absent'
            ])

        return response
    export_attendance_report.short_description = "Export selected records as CSV"



admin_site.register(Student, StudentAdmin)
admin_site.register(Course, CourseAdmin)
admin_site.register(Attendance, AttendanceAdmin)
admin_site.register(UserCourse, UserCourseAdmin)

admin_site.register(User, UserAdmin)
admin_site.register(Group, GroupAdmin)