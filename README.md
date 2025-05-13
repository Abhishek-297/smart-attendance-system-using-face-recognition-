# Smart Attendance System Using Face Recognition

A web-based attendance system built with Django and OpenCV that uses face recognition to mark attendance automatically. It streamlines traditional attendance processes by integrating real-time face detection and recognition.

## 🔧 Features

- 🎓 Manage courses, subjects, and students.
- 📸 Real-time face recognition using webcam.
- 🗓️ Automatic attendance marking.
- 📊 Admin dashboard with attendance summaries.
- 🧠 Uses `face_recognition` and OpenCV for accuracy.
- 🔒 Secure login for Admin, Faculty, and Students.
- 📂 Django-based backend with MySQL integration.

## 📁 Project Structure
smart_attendance/

├── attendance/ # Django app for attendance logic
├── smart_attendance/ # Django project settings
├── static/ # Static files (CSS, JS, images)
├── media/ # For storing student faces
├── templates/ # HTML templates
├── utils/ # Face recognition logic
├── db.sqlite3 # Default SQLite DB ( I used MySQL)
└── manage.py # Django management script


## ⚙️ Technologies Used

- Python
- Django
- OpenCV
- face_recognition
- MySQL (or SQLite)
- HTML, CSS, Bootstrap
- JavaScript

## 🚀 Setup Instructions

1. Clone the repository
              git clone https://github.com/yourusername/smart_attendance.git
   
2. Create a virtual environment and install dependencies
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   
3. Configure the database
   Edit settings.py for your preferred database (MySQL). Create the database and apply migrations:
   python manage.py makemigrations
   python manage.py migrate
   
4. Create a superuser
   python manage.py createsuperuser

5. Run the development server
   python manage.py runserver
   Visit http://127.0.0.1:8000 in your browser.


## 📸 Face Recognition Usage

Upload student images via the admin dashboard or capture directly from webcam.

The face embeddings are stored and used for real-time recognition during attendance.

## 🛠️ Utilities

utils/face_recognition.py handles face encoding and recognition.

Webcam or camera access is done through JavaScript  in html files and OpenCV integration.

## 📌 Notes
Ensure your webcam or camera   is working.

Use clear and front-facing images for best results.

Haar Cascade is used as fallback detection.

## 🧑‍💻 Contributing

Pull requests are welcome. For major changes, please open an issue first.

## 📃 License

MIT License






