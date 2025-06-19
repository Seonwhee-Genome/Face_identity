import os
import django

# Set up Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'facetest.settings')
django.setup()

from vectorstore.models import Searchmanager

def compute_f1(user_name):
    # Count TP, FP, FN
    tp = Searchmanager.objects.filter(user=user_name, identify=True, correct=True).count()
    fp = Searchmanager.objects.filter(user=user_name, identify=True, correct=False).count()
    fn = Searchmanager.objects.filter(user=user_name, identify=False, correct=False).count()

    # Compute metrics
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    print(f"[User: {user_name}] TP: {tp}, FP: {fp}, FN: {fn}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}, Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    compute_f1("dki1236")  # ← Replace with actual user name
