import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import time
from functools import wraps
from textwrap import dedent


def send_email_notification(
    to_email,
    password,
    from_email=None,
    email_subject=None,
    task_name=None,
    smtp_server="smtp.gmail.com",
    smtp_port=587,
):
    """
    Decorator: Sends email notification after function execution completes

    Args:
        to_email: Email address to receive the notification
        password: Password for the sending email account (app-specific password)
        from_email: Email address to send from (defaults to same as to_email)
        email_subject: Subject line for the email
        task_name: Name of the task being executed
        smtp_server: SMTP server address (default Gmail)
        smtp_port: SMTP server port (default 587)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Record start time
            start_time = time.time()

            # Execute training
            func(*args, **kwargs)

            # Calculate execution time
            execution_time = time.time() - start_time

            _task_name = task_name if task_name else "Training Task"

            _email_subject = (
                email_subject
                if email_subject
                else f"Training Complete Notification - {_task_name}"
            )

            body = dedent(f"""
                Training complete!

                Task name: {_task_name}
                Execution time: 
                    {execution_time / 3600:.2f} hours 
                    {execution_time / 60:.2f} minutes 
                    {execution_time:.2f} seconds
                Execution status: Success
            """)
            _send_email(
                email_subject=_email_subject,
                body=body,
                to_email=to_email,
                from_email=from_email or to_email,
                password=password,
                smtp_server=smtp_server,
                smtp_port=smtp_port,
            )

        return wrapper

    return decorator


def _send_email(
    email_subject,
    body,
    to_email,
    from_email,
    password,
    smtp_server,
    smtp_port,
):
    """Internal function: Send email"""
    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = to_email
    msg["Subject"] = email_subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(from_email, password)
        server.send_message(msg)
        server.quit()
        print(f"Email notification sent to {to_email}")
    except Exception as e:
        print(f"Email sending failed: {e}")


@send_email_notification(
    to_email="xujialiu@link.cuhk.edu.hk",
    password="kkkk kkkk kkkk kkkk",
    from_email="xujialiuphd@gmail.com",
)
def train_model():
    """Model training function"""
    for i in range(10):
        print(f"Training epoch {i + 1}...")
        time.sleep(0.1)  # Simulate training process
    print("Training complete!")


if __name__ == "__main__":
    train_model()